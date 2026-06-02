"""
backend/main_api.py
-------------------
FastAPI + WebSocket para PathForge.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from backend.core.constraints import (
    Constraint, ConstraintProfiles,
    MaxRiskConstraint, MaxYearsConstraint, MinSalaryConstraint,
)
from backend.core.generator import GeneratorConfig, TrajectoryGenerator
from backend.core.graph import CareerGraph
from backend.core.simulation import CareerSimulator
from backend.core.scorer import CareerOutcomePredictor
from backend.data.loader import load_career_graph
from backend.data.input_manager import InputManager, UserInput
from backend.llm.analyzer import TrajectoryAnalyzer
from backend.llm.client import get_llm_client

# ──────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="PathForge API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ──────────────────────────────────────────────────────────────
# Estado global — inicializado una vez
# ──────────────────────────────────────────────────────────────

_career_graph: CareerGraph | None = None
_predictor:    CareerOutcomePredictor | None = None
_simulator:    CareerSimulator | None = None


def _group_by_terminal(results: list) -> dict[str, list]:
    """Agrupa trayectorias por nodo terminal final."""
    groups: dict[str, list] = {}
    for et in results:
        terminal = et.trajectory.nodes[-1] if et.trajectory.nodes else "unknown"
        groups.setdefault(terminal, []).append(et)
    return groups


def _init_components():
    """Inicializa todos los componentes de IA en orden correcto."""
    global _career_graph, _predictor, _simulator

    if _career_graph is not None:
        return

    logger.info("Inicializando componentes de IA...")

    # 1. Entrenar / cargar modelo sklearn
    _predictor = CareerOutcomePredictor.load_or_train()

    # 2. Inicializar simulador Monte Carlo
    _simulator = CareerSimulator(n_simulations=40, outcome_model=None)

    # 3. Cargar grafo con predictor integrado
    raw_graph    = load_career_graph()
    _career_graph = CareerGraph(raw_graph, outcome_predictor=_predictor)

    logger.success(f"Sistema listo | {_career_graph}")


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _init_components)


def get_graph() -> CareerGraph:
    _init_components()
    return _career_graph


# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────

class NodeInput(BaseModel):
    id: str
    label: str
    avg_salary: float = Field(gt=0)
    demand: float = Field(ge=0, le=1)
    satisfaction: float = Field(ge=0, le=1)
    years_experience: int = Field(ge=0)
    skills: list[str] = []
    type: str = "role"
    is_terminal: bool = False


class EdgeInput(BaseModel):
    from_node: str
    to_node: str
    transition_years: int = Field(ge=0)
    difficulty: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)


class ExploreRequest(BaseModel):
    source: str
    nodes: list[NodeInput] = []
    edges: list[EdgeInput] = []
    profile: str = "balanced"
    max_years: int = 12
    max_risk: float = 0.6
    beam_width: int = 10
    max_depth: int = 6
    top_k: int = 15
    user_profile: str = "profesional de tecnología"
    use_simulation: bool = True


class AnalyzeRequest(BaseModel):
    trajectories: list[dict]
    criterion: str = "mejor trayectoria general"
    user_profile: str = "profesional de tecnología"


class SimulateRequest(BaseModel):
    path: list[str]
    n_simulations: int = Field(default=50, ge=10, le=200)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def build_constraint(req: ExploreRequest) -> Constraint:
    profiles = {
        "conservative": ConstraintProfiles.conservative(),
        "ambitious":    ConstraintProfiles.ambitious(),
        "balanced":     ConstraintProfiles.balanced(req.max_years),
        "fast_track":   ConstraintProfiles.fast_track(),
    }
    base = profiles.get(req.profile, ConstraintProfiles.balanced())
    return base & MaxRiskConstraint(req.max_risk) & MaxYearsConstraint(req.max_years)


def build_graph_from_request(req: ExploreRequest) -> CareerGraph:
    if req.nodes and req.edges:
        import networkx as nx
        G = nx.DiGraph()
        for n in req.nodes:
            G.add_node(n.id, **n.model_dump())
        for e in req.edges:
            G.add_edge(
                e.from_node, e.to_node,
                transition_years=e.transition_years,
                difficulty=e.difficulty,
                risk=e.risk,
            )
        return CareerGraph(G, outcome_predictor=_predictor)
    return get_graph()


def traj_to_dict(et) -> dict:
    return {
        "nodes":             list(et.trajectory.nodes),
        "scores":            et.scores,
        "pareto_rank":       et.pareto_rank,
        "crowding_distance": et.crowding_distance if et.crowding_distance != float("inf") else 9999.0,
        "terminal_node":     et.trajectory.nodes[-1] if et.trajectory.nodes else "",
        "is_terminal_end":   et.scores.get("is_terminal_end", 0.0) > 0.5,
    }


# ──────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    return FileResponse(str(index)) if index.exists() else {"message": "PathForge API v3.0"}


@app.get("/api/graph")
async def get_default_graph():
    graph = get_graph()
    nodes = [{"id": nid, **graph.node_attrs(nid)} for nid in graph.all_node_ids()]
    edges = [{"from": u, "to": v, **graph.edge_attrs(u, v)} for u, v in graph._g.edges()]
    return {
        "nodes":     nodes,
        "edges":     edges,
        "terminals": graph.terminal_nodes(),
    }


@app.get("/api/terminals")
async def get_terminals():
    """Retorna los nodos terminales (posibles finales de trayectoria)."""
    graph = get_graph()
    terminals = graph.terminal_nodes()
    return {
        "terminals": [
            {"id": tid, **graph.node_attrs(tid)}
            for tid in terminals
        ]
    }


@app.get("/api/model/info")
async def model_info():
    """Información del modelo sklearn."""
    if _predictor is None:
        return {"available": False}
    return {
        "available":            True,
        "cv_auc":               round(_predictor.cv_score, 4),
        "feature_importances":  _predictor.feature_importances,
        "model_type":           "GradientBoostingClassifier",
    }


@app.post("/api/generate")
async def generate_trajectories(req: ExploreRequest):
    graph       = build_graph_from_request(req)
    constraints = build_constraint(req)
    config      = GeneratorConfig(
        beam_width=req.beam_width, max_depth=req.max_depth,
        top_k_results=req.top_k,
    )
    generator = TrajectoryGenerator(graph, config)
    results   = generator.generate(req.source, constraints)
    groups    = _group_by_terminal(results)

    return {
        "trajectories":    [traj_to_dict(et) for et in results],
        "terminal_groups": {k: [traj_to_dict(et) for et in v] for k, v in groups.items()},
        "terminals_found": list(groups.keys()),
    }


@app.post("/api/simulate")
async def simulate_trajectory(req: SimulateRequest):
    """Ejecuta simulación Monte Carlo en una trayectoria específica."""
    if _simulator is None:
        return {"error": "Simulator not initialized"}

    graph = get_graph()
    node_attrs = {nid: graph.node_attrs(nid) for nid in graph.all_node_ids()}
    edge_attrs = {(u, v): graph.edge_attrs(u, v) for u, v in graph._g.edges()}

    sim = CareerSimulator(n_simulations=req.n_simulations)
    result = sim.monte_carlo(tuple(req.path), node_attrs, edge_attrs)
    return result


@app.get("/api/llm/status")
async def llm_status():
    try:
        client = get_llm_client()
        return {"key_count": client.key_count, "active_provider": client.active_provider, "keys": client.status()}
    except Exception as e:
        return {"error": str(e), "key_count": 0}


@app.post("/api/analyze")
async def analyze_trajectories(req: AnalyzeRequest):
    """Análisis cualitativo de trayectorias con LLM (ahora en español)."""
    try:
        from backend.core.evaluator import EvaluatedTrajectory
        from backend.core.graph import Trajectory

        evaluated = []
        for t in req.trajectories:
            traj = Trajectory(nodes=tuple(t["nodes"]))
            et   = EvaluatedTrajectory(
                trajectory=traj, scores=t["scores"],
                pareto_rank=t["pareto_rank"],
                crowding_distance=t.get("crowding_distance", 0.0),
            )
            evaluated.append(et)

        analyzer = TrajectoryAnalyzer(user_profile=req.user_profile)
        result   = analyzer.rank_by(evaluated, req.criterion)
        return {
            "analysis":              result.content,
            "provider_used":         result.provider_used,
            "trajectories_analyzed": result.trajectories_analyzed,
        }
    except Exception as exc:
        logger.error(f"Error en análisis: {exc}")
        return {"analysis": f"Error: {exc}", "trajectories_analyzed": 0}


# ──────────────────────────────────────────────────────────────
# Gestión de Inputs de Usuario (BD SQLite)
# ──────────────────────────────────────────────────────────────

class UserInputRequest(BaseModel):
    id: str
    source_career: str
    profile: str = "balanced"
    max_years: int = 12
    max_risk: float = 0.6
    beam_width: int = 10
    max_depth: int = 6
    top_k: int = 15
    user_profile_description: str = "profesional de tecnología"
    notes: str = ""


@app.post("/api/inputs/create")
async def create_input(req: UserInputRequest):
    """Crear y guardar nueva entrada."""
    try:
        manager = InputManager()
        user_input = UserInput(
            id=req.id,
            source_career=req.source_career,
            profile=req.profile,
            max_years=req.max_years,
            max_risk=req.max_risk,
            beam_width=req.beam_width,
            max_depth=req.max_depth,
            top_k=req.top_k,
            user_profile_description=req.user_profile_description,
            notes=req.notes,
        )
        manager.save_input(user_input)
        return {"success": True, "id": user_input.id}
    except Exception as e:
        logger.error(f"Error creando input: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/inputs/list")
async def list_inputs():
    """Listar todos los inputs guardados."""
    try:
        manager = InputManager()
        inputs = manager.list_inputs()
        return {
            "inputs": [
                {
                    "id": inp.id,
                    "source_career": inp.source_career,
                    "profile": inp.profile,
                    "max_years": inp.max_years,
                    "max_risk": inp.max_risk,
                    "user_profile_description": inp.user_profile_description,
                    "created_at": inp.created_at,
                    "updated_at": inp.updated_at,
                }
                for inp in inputs
            ]
        }
    except Exception as e:
        logger.error(f"Error listando inputs: {e}")
        return {"inputs": [], "error": str(e)}


@app.get("/api/inputs/{input_id}")
async def get_input(input_id: str):
    """Obtener detalles de un input específico."""
    try:
        manager = InputManager()
        user_input = manager.load_input(input_id)
        if not user_input:
            return {"error": "Input not found"}
        return {
            "id": user_input.id,
            "source_career": user_input.source_career,
            "profile": user_input.profile,
            "max_years": user_input.max_years,
            "max_risk": user_input.max_risk,
            "beam_width": user_input.beam_width,
            "max_depth": user_input.max_depth,
            "top_k": user_input.top_k,
            "user_profile_description": user_input.user_profile_description,
            "notes": user_input.notes,
            "created_at": user_input.created_at,
            "updated_at": user_input.updated_at,
        }
    except Exception as e:
        logger.error(f"Error obteniendo input: {e}")
        return {"error": str(e)}


@app.delete("/api/inputs/{input_id}")
async def delete_input(input_id: str):
    """Eliminar un input."""
    try:
        manager = InputManager()
        manager.delete_input(input_id)
        return {"success": True}
    except Exception as e:
        logger.error(f"Error eliminando input: {e}")
        return {"success": False, "error": str(e)}


# ──────────────────────────────────────────────────────────────
# WebSocket — stream enriquecido del Beam Search
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/explore")
async def websocket_explore(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket conectado")

    try:
        raw     = await websocket.receive_text()
        message = json.loads(raw)
        if message.get("type") != "start":
            await websocket.send_json({"type": "error", "msg": "Esperaba type=start"})
            return

        req         = ExploreRequest(**message["data"])
        graph       = build_graph_from_request(req)
        constraints = build_constraint(req)
        config      = GeneratorConfig(
            beam_width=req.beam_width, max_depth=req.max_depth,
            top_k_results=req.top_k, emit_steps=True,
        )
        generator = TrajectoryGenerator(graph, config)
        loop      = asyncio.get_event_loop()

        # Emitir información del grafo al inicio
        await websocket.send_json({
            "type":      "graph_info",
            "nodes":     [{"id": nid, **graph.node_attrs(nid)} for nid in graph.all_node_ids()],
            "edges":     [{"from": u, "to": v, **graph.edge_attrs(u, v)} for u, v in graph._g.edges()],
            "terminals": graph.terminal_nodes(),
            "source":    req.source,
        })

        async def emit_step(depth: int, beam: list, completed: list):
            await websocket.send_json({
                "type": "step",
                "depth": depth,
                "beam": [list(p) for p in beam],
                "completed": [list(p) for p in completed],
            })
            await asyncio.sleep(0.06)  # pausa para animación fluida

        def sync_callback(depth: int, beam: list, completed: list):
            asyncio.run_coroutine_threadsafe(emit_step(depth, beam, completed), loop)

        results = await loop.run_in_executor(
            None,
            lambda: generator.generate(
                source=req.source,
                constraints=constraints,
                step_callback=sync_callback,
            ),
        )

        groups = _group_by_terminal(results)

        await websocket.send_json({
            "type":            "result",
            "trajectories":    [traj_to_dict(et) for et in results],
            "terminal_groups": {k: [traj_to_dict(et) for et in v] for k, v in groups.items()},
            "terminals_found": list(groups.keys()),
        })
        await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket desconectado")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")
        try:
            await websocket.send_json({"type": "error", "msg": str(exc)})
        except Exception:
            pass


if __name__ == "__main__":
    import os
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main_api:app", host=host, port=port, reload=True)