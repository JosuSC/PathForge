"""
data/loader.py
--------------
Responsabilidad única: cargar y validar el dataset de carreras desde disco.
Expone el grafo como estructura NetworkX lista para usar por el core.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Modelos de validación (Pydantic v2)
# ---------------------------------------------------------------------------

class CareerNode(BaseModel):
    """Representa un rol o posición profesional."""

    id: str
    label: str
    type: str
    skills: list[str]
    avg_salary: float = Field(gt=0)
    demand: float = Field(ge=0.0, le=1.0)
    satisfaction: float = Field(ge=0.0, le=1.0)
    years_experience: int = Field(ge=0)

    @field_validator("skills")
    @classmethod
    def skills_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Un nodo debe tener al menos una habilidad.")
        return v


class CareerEdge(BaseModel):
    """Representa una transición posible entre dos roles."""

    from_: str = Field(alias="from")
    to: str
    transition_years: int = Field(ge=0)
    difficulty: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class CareerDataset(BaseModel):
    """Schema completo del dataset."""

    nodes: list[CareerNode]
    edges: list[CareerEdge]


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def load_career_graph(path: str | Path | None = None) -> nx.DiGraph:
    """
    Carga el dataset desde JSON y construye un grafo dirigido NetworkX.

    Args:
        path: Ruta al archivo JSON. Por defecto usa careers.json en la misma carpeta.

    Returns:
        nx.DiGraph con nodos y aristas enriquecidos con sus atributos.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValidationError: Si el JSON no cumple el schema.
    """
    if path is None:
        path = Path(__file__).parent / "careers.json"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    logger.info(f"Cargando dataset desde {path}")

    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    dataset = CareerDataset(**raw)

    graph = _build_graph(dataset)

    logger.success(
        f"Grafo cargado: {graph.number_of_nodes()} nodos, "
        f"{graph.number_of_edges()} aristas"
    )
    return graph


def _build_graph(dataset: CareerDataset) -> nx.DiGraph:
    """Construye el DiGraph a partir del dataset validado."""
    G = nx.DiGraph()

    for node in dataset.nodes:
        G.add_node(node.id, **node.model_dump())

    for edge in dataset.edges:
        G.add_edge(
            edge.from_,
            edge.to,
            transition_years=edge.transition_years,
            difficulty=edge.difficulty,
            risk=edge.risk,
        )

    return G