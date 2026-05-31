"""
backend/main_api.py
-------------------
Servidor FastAPI + WebSocket para PathForge.
Sirve el frontend estático y expone los endpoints de la API.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from backend.core.constraints import (
    Constraint,
    ConstraintProfiles,
    MaxRiskConstraint,
    MaxYearsConstraint,
    MinSalaryConstraint,
)
from backend.core.generator import GeneratorConfig, TrajectoryGenerator
from backend.core.graph import CareerGraph
from backend.data.loader import load_career_graph
from backend.llm.analyzer import TrajectoryAnalyzer
from backend.llm.client import get_llm_client

# ──────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="PathForge API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos del frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ──────────────────────────────────────────────────────────────
# Estado global
# ──────────────────────────────────────────────────────────────

_career_graph: CareerGraph | None = None


def get_graph() -> CareerGraph:
    global _career_graph
    if _career_graph is None:
        _career_graph = CareerGraph(load_career_graph())
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
        return CareerGraph(G)
    return get_graph()


def traj_to_dict(et) -> dict:
    return {
        "nodes": list(et.trajectory.nodes),
        "scores": et.scores,
        "pareto_rank": et.pareto_rank,
        "crowding_distance": (
            et.crowding_distance if et.crowding_distance != float("inf") else 9999.0
        ),
    }


# ──────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "PathForge API v2.0", "docs": "/docs"}


@app.get("/api/graph")
async def get_default_graph():
    graph = get_graph()
    nodes = [{"id": nid, **graph.node_attrs(nid)} for nid in graph.all_node_ids()]
    edges = [
        {"from": u, "to": v, **graph.edge_attrs(u, v)}
        for u, v in graph._g.edges()
    ]
    return {"nodes": nodes, "edges": edges}


@app.get("/api/llm/status")
async def llm_status():
    """Estado de todas las API keys en la cola circular."""
    try:
        client = get_llm_client()
        return {
            "key_count": client.key_count,
            "active_provider": client.active_provider,
            "keys": client.status(),
        }
    except Exception as e:
        return {"error": str(e), "key_count": 0}


@app.post("/api/generate")
async def generate_trajectories(req: ExploreRequest):
    graph = build_graph_from_request(req)
    constraints = build_constraint(req)
    config = GeneratorConfig(
        beam_width=req.beam_width,
        max_depth=req.max_depth,
        top_k_results=req.top_k,
    )
    generator = TrajectoryGenerator(graph, config)
    results = generator.generate(req.source, constraints)
    return {"trajectories": [traj_to_dict(et) for et in results]}


@app.post("/api/analyze")
async def analyze_trajectories(req: AnalyzeRequest):
    try:
        from backend.core.evaluator import EvaluatedTrajectory
        from backend.core.graph import Trajectory

        evaluated = []
        for t in req.trajectories:
            traj = Trajectory(nodes=tuple(t["nodes"]))
            et = EvaluatedTrajectory(
                trajectory=traj,
                scores=t["scores"],
                pareto_rank=t["pareto_rank"],
                crowding_distance=t.get("crowding_distance", 0.0),
            )
            evaluated.append(et)

        analyzer = TrajectoryAnalyzer(user_profile=req.user_profile)
        result = analyzer.rank_by(evaluated, req.criterion)

        return {
            "analysis": result.content,
            "provider_used": result.provider_used,
            "trajectories_analyzed": result.trajectories_analyzed,
        }
    except Exception as exc:
        logger.error(f"Error en análisis: {exc}")
        return {"analysis": f"Error: {exc}", "trajectories_analyzed": 0}


# ──────────────────────────────────────────────────────────────
# WebSocket
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/explore")
async def websocket_explore(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket conectado")
    try:
        raw = await websocket.receive_text()
        message = json.loads(raw)
        if message.get("type") != "start":
            await websocket.send_json({"type": "error", "msg": "Esperaba type=start"})
            return

        req = ExploreRequest(**message["data"])
        graph = build_graph_from_request(req)
        constraints = build_constraint(req)
        config = GeneratorConfig(
            beam_width=req.beam_width,
            max_depth=req.max_depth,
            top_k_results=req.top_k,
            emit_steps=True,
        )
        generator = TrajectoryGenerator(graph, config)
        loop = asyncio.get_event_loop()

        async def emit_step(depth: int, beam: list, completed: list):
            await websocket.send_json({
                "type": "step",
                "depth": depth,
                "beam": [list(p) for p in beam],
                "completed": [list(p) for p in completed[-20:]],
            })
            await asyncio.sleep(0.04)

        def sync_callback(depth, beam, completed):
            asyncio.run_coroutine_threadsafe(
                emit_step(depth, beam, completed), loop
            )

        results = await loop.run_in_executor(
            None,
            lambda: generator.generate(
                source=req.source,
                constraints=constraints,
                step_callback=sync_callback,
            ),
        )

        await websocket.send_json({
            "type": "result",
            "trajectories": [traj_to_dict(et) for et in results],
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


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main_api:app", host=host, port=port, reload=True)