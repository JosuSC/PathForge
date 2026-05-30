"""
main_api.py
-----------
Servidor FastAPI con WebSocket para PathForge.

Endpoints:
    GET  /api/graph          → Retorna el grafo completo (nodos + aristas)
    POST /api/generate       → Genera trayectorias con configuración custom
    WS   /ws/explore         → Stream en tiempo real del Beam Search
    POST /api/analyze        → Análisis cualitativo con Gemini
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from core.constraints import (
    Constraint,
    ConstraintProfiles,
    MaxRiskConstraint,
    MaxYearsConstraint,
    MinSalaryConstraint,
)
from core.generator import GeneratorConfig, TrajectoryGenerator
from core.graph import CareerGraph
from data.loader import load_career_graph
from llm.analyzer import TrajectoryAnalyzer

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="PathForge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir frontend estático
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# ---------------------------------------------------------------------------
# Estado global (cargado una vez al inicio)
# ---------------------------------------------------------------------------

_career_graph: CareerGraph | None = None


def get_graph() -> CareerGraph:
    global _career_graph
    if _career_graph is None:
        _career_graph = CareerGraph(load_career_graph())
    return _career_graph


# ---------------------------------------------------------------------------
# Schemas de entrada/salida
# ---------------------------------------------------------------------------

class NodeInput(BaseModel):
    id: str
    label: str
    avg_salary: float = Field(gt=0)
    demand: float = Field(ge=0, le=1)
    satisfaction: float = Field(ge=0, le=1)
    years_experience: int = Field(ge=0)
    skills: list[str] = []
    type: str = "role"


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


class AnalyzeRequest(BaseModel):
    trajectories: list[dict]
    criterion: str = "mejor trayectoria general"
    user_profile: str = "profesional de tecnología"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_constraint(request: ExploreRequest) -> Constraint:
    """Construye restricción combinada desde los parámetros del request."""
    profiles = {
        "conservative": ConstraintProfiles.conservative(),
        "ambitious":    ConstraintProfiles.ambitious(),
        "balanced":     ConstraintProfiles.balanced(request.max_years),
        "fast_track":   ConstraintProfiles.fast_track(),
    }
    base = profiles.get(request.profile, ConstraintProfiles.balanced())
    extra = MaxRiskConstraint(request.max_risk) & MaxYearsConstraint(request.max_years)
    return base & extra


def build_graph_from_request(request: ExploreRequest) -> CareerGraph:
    """
    Si el request incluye nodos/aristas personalizados, construye un grafo
    desde ellos. Si no, usa el grafo por defecto.
    """
    if request.nodes and request.edges:
        import networkx as nx
        G = nx.DiGraph()
        for n in request.nodes:
            G.add_node(n.id, **n.model_dump())
        for e in request.edges:
            G.add_edge(
                e.from_node, e.to_node,
                transition_years=e.transition_years,
                difficulty=e.difficulty,
                risk=e.risk,
            )
        return CareerGraph(G)
    return get_graph()


def trajectory_to_dict(et) -> dict:
    """Serializa EvaluatedTrajectory a dict JSON-friendly."""
    return {
        "nodes": list(et.trajectory.nodes),
        "scores": et.scores,
        "pareto_rank": et.pareto_rank,
        "crowding_distance": (
            et.crowding_distance
            if et.crowding_distance != float("inf")
            else 9999.0
        ),
    }


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    return FileResponse("../frontend/index.html")


@app.get("/api/graph")
async def get_default_graph():
    """Retorna el grafo por defecto como JSON para renderizar en 3D."""
    graph = get_graph()
    nodes = [
        {"id": nid, **graph.node_attrs(nid)}
        for nid in graph.all_node_ids()
    ]
    edges = [
        {"from": u, "to": v, **graph.edge_attrs(u, v)}
        for u, v in graph._g.edges()
    ]
    return {"nodes": nodes, "edges": edges}


@app.post("/api/generate")
async def generate_trajectories(request: ExploreRequest):
    """Genera trayectorias de forma síncrona (sin animación)."""
    graph = build_graph_from_request(request)
    constraints = build_constraint(request)
    config = GeneratorConfig(
        beam_width=request.beam_width,
        max_depth=request.max_depth,
        top_k_results=request.top_k,
    )
    generator = TrajectoryGenerator(graph, config)
    results = generator.generate(request.source, constraints)
    return {"trajectories": [trajectory_to_dict(et) for et in results]}


@app.post("/api/analyze")
async def analyze_trajectories(request: AnalyzeRequest):
    """Análisis cualitativo con Gemini."""
    try:
        from core.evaluator import EvaluatedTrajectory
        from core.graph import Trajectory

        # Reconstruir objetos desde los dicts
        evaluated = []
        for t in request.trajectories:
            traj = Trajectory(nodes=tuple(t["nodes"]))
            et = EvaluatedTrajectory(
                trajectory=traj,
                scores=t["scores"],
                pareto_rank=t["pareto_rank"],
                crowding_distance=t.get("crowding_distance", 0.0),
            )
            evaluated.append(et)

        analyzer = TrajectoryAnalyzer(user_profile=request.user_profile)
        result = analyzer.rank_by(evaluated, request.criterion)

        return {
            "analysis": result.content,
            "trajectories_analyzed": result.trajectories_analyzed,
        }
    except Exception as exc:
        logger.error(f"Error en análisis: {exc}")
        return {"analysis": f"Error al consultar Gemini: {exc}", "trajectories_analyzed": 0}


# ---------------------------------------------------------------------------
# WebSocket — Stream del Beam Search en tiempo real
# ---------------------------------------------------------------------------

@app.websocket("/ws/explore")
async def websocket_explore(websocket: WebSocket):
    """
    WebSocket que emite cada paso del Beam Search en tiempo real.

    Protocolo:
        Cliente → {"type": "start", "data": ExploreRequest}
        Servidor → {"type": "step", "depth": N, "beam": [...], "completed": [...]}
        Servidor → {"type": "result", "trajectories": [...]}
        Servidor → {"type": "done"}
    """
    await websocket.accept()
    logger.info("WebSocket conectado")

    try:
        raw = await websocket.receive_text()
        message = json.loads(raw)

        if message.get("type") != "start":
            await websocket.send_json({"type": "error", "msg": "Esperaba type=start"})
            return

        request = ExploreRequest(**message["data"])
        graph = build_graph_from_request(request)
        constraints = build_constraint(request)

        config = GeneratorConfig(
            beam_width=request.beam_width,
            max_depth=request.max_depth,
            top_k_results=request.top_k,
            emit_steps=True,
        )

        generator = TrajectoryGenerator(graph, config)
        results_holder: list = []

        async def emit_step(depth: int, beam: list, completed: list):
            """Callback async que emite cada paso del beam al cliente."""
            await websocket.send_json({
                "type": "step",
                "depth": depth,
                "beam": [list(p) for p in beam],
                "completed": [list(p) for p in completed[-20:]],  # últimas 20
            })
            await asyncio.sleep(0.05)  # pausa visual para la animación

        # Ejecutar en thread separado para no bloquear el event loop
        loop = asyncio.get_event_loop()

        def sync_callback(depth, beam, completed):
            asyncio.run_coroutine_threadsafe(
                emit_step(depth, beam, completed), loop
            )

        results = await loop.run_in_executor(
            None,
            lambda: generator.generate(
                source=request.source,
                constraints=constraints,
                step_callback=sync_callback,
            ),
        )

        # Enviar resultados finales
        await websocket.send_json({
            "type": "result",
            "trajectories": [trajectory_to_dict(et) for et in results],
        })
        await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket desconectado")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")
        await websocket.send_json({"type": "error", "msg": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
