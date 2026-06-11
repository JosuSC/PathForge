"""
data/loader.py
--------------
Responsabilidad única: cargar y validar el dataset de carreras desde disco.
Expone el grafo como estructura NetworkX lista para usar por el core.

- CareerEdge ahora incluye transition_probability y salary_growth (opcionales)
- _build_graph pasa esos atributos al DiGraph en lugar de ignorarlos
- _build_graph valida referencias: aristas con nodos inexistentes se descartan con warning
- list_available_domains() lee la clave 'domain' de metadata.json (no 'sector')
- CareerNode.type acepta todos los valores reales del sistema
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

import networkx as nx
from loguru import logger
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Modelos de validación (Pydantic v2)
# ---------------------------------------------------------------------------

# FIX [03]: type acepta todos los valores que genera transform_data + el legacy 'role'
NODE_TYPES = Literal["entry", "mid", "senior", "leadership", "role"]


class CareerNode(BaseModel):
    """Representa un rol o posición profesional."""

    id: str
    label: str
    type: NODE_TYPES
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
    # FIX [02]: campos generados por transform_data — opcionales para compatibilidad
    # con careers.json legacy que no los tiene aún
    transition_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    salary_growth: Optional[float] = Field(default=None, ge=0.0)

    model_config = {"populate_by_name": True}


class CareerDataset(BaseModel):
    """Schema completo del dataset."""

    nodes: list[CareerNode]
    edges: list[CareerEdge]


# ---------------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent
PROBLEMS_DIR = DATA_DIR / "problems"


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
        path = DATA_DIR / "careers.json"

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


def list_available_domains() -> list[dict[str, Any]]:
    """
    Lista todos los dominios de carrera disponibles en problems/.

    Soporta dos formatos:
    - Archivo plano: problems/technology.json
    - Directorio:    problems/software_development/graph.json + metadata.json

    Returns:
        Lista de diccionarios con info de cada dominio.
    """
    domains = []

    if not PROBLEMS_DIR.exists():
        return domains

    # 1. Archivos JSON planos (formato original / legacy)
    for json_file in sorted(PROBLEMS_DIR.glob("*.json")):
        if json_file.name == "test_instances.json":
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if "nodes" in data and "edges" in data:
                domains.append({
                    "id": json_file.stem,
                    "name": json_file.stem.replace("_", " ").title(),
                    "format": "flat",
                    "path": str(json_file.relative_to(DATA_DIR)),
                    "nodes": len(data.get("nodes", [])),
                    "edges": len(data.get("edges", [])),
                })
        except Exception:
            pass

    # 2. Directorios con graph.json (formato transform_data.py)
    for domain_dir in sorted(PROBLEMS_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        graph_file = domain_dir / "graph.json"
        metadata_file = domain_dir / "metadata.json"
        if not graph_file.exists():
            continue

        meta: dict[str, Any] = {}
        if metadata_file.exists():
            try:
                meta = json.loads(metadata_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        try:
            gdata = json.loads(graph_file.read_text(encoding="utf-8"))
            n_nodes = len(gdata.get("nodes", []))
            n_edges = len(gdata.get("edges", []))
        except Exception:
            n_nodes, n_edges = 0, 0

        # FIX [01]: la clave correcta en metadata.json es 'domain', no 'sector'
        domain_name = (
            meta.get("domain")
            or meta.get("sector")  # compatibilidad con metadata legacy
            or domain_dir.name.replace("_", " ").title()
        )

        domains.append({
            "id": domain_dir.name,
            "name": domain_name.replace("_", " ").title() if "_" in domain_name else domain_name,
            "format": "directory",
            "path": str(graph_file.relative_to(DATA_DIR)),
            "nodes": n_nodes,
            "edges": n_edges,
            "broad_sector": meta.get("broad_sector", ""),
            "description": meta.get("description", ""),
            "source": meta.get("source", ""),
            "occupations": meta.get("occupations", []),
        })

    return domains


def load_domain_graph(domain_id: str) -> nx.DiGraph:
    """
    Carga el grafo de un dominio específico desde problems/.

    Args:
        domain_id: ID del dominio (ej: 'software_development')

    Returns:
        nx.DiGraph con nodos y aristas del dominio.

    Raises:
        FileNotFoundError: Si el dominio no existe.
    """
    # Formato plano primero
    flat_path = PROBLEMS_DIR / f"{domain_id}.json"
    if flat_path.exists():
        return load_career_graph(flat_path)

    # Formato directorio
    dir_path = PROBLEMS_DIR / domain_id / "graph.json"
    if dir_path.exists():
        return load_career_graph(dir_path)

    raise FileNotFoundError(
        f"Dominio '{domain_id}' no encontrado. "
        f"Buscado en: {flat_path} y {dir_path}"
    )


def _build_graph(dataset: CareerDataset) -> nx.DiGraph:
    """Construye el DiGraph a partir del dataset validado."""
    G = nx.DiGraph()

    # Añadir nodos
    node_ids: set[str] = set()
    for node in dataset.nodes:
        G.add_node(node.id, **node.model_dump())
        node_ids.add(node.id)

    # FIX [09]: validar que los extremos de cada arista existan como nodos
    skipped = 0
    for edge in dataset.edges:
        if edge.from_ not in node_ids:
            logger.warning(
                f"Arista descartada: nodo origen '{edge.from_}' no existe en el grafo."
            )
            skipped += 1
            continue
        if edge.to not in node_ids:
            logger.warning(
                f"Arista descartada: nodo destino '{edge.to}' no existe en el grafo."
            )
            skipped += 1
            continue

        # FIX [02]: pasar transition_probability y salary_growth si están presentes
        edge_attrs: dict[str, Any] = {
            "transition_years": edge.transition_years,
            "difficulty": edge.difficulty,
            "risk": edge.risk,
        }
        if edge.transition_probability is not None:
            edge_attrs["transition_probability"] = edge.transition_probability
        if edge.salary_growth is not None:
            edge_attrs["salary_growth"] = edge.salary_growth

        G.add_edge(edge.from_, edge.to, **edge_attrs)

    if skipped:
        logger.warning(f"{skipped} aristas descartadas por referencias inválidas.")

    return G
