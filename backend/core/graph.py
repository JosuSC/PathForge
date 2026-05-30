"""
core/graph.py
-------------
Capa de abstracción sobre el grafo de carreras.
Ofrece operaciones de alto nivel: vecinos, validación de caminos,
cálculo de atributos agregados de una trayectoria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Trajectory:
    """
    Representa una trayectoria profesional como secuencia de IDs de nodos.

    Inmutable (frozen) para poder usarla como clave de diccionario y en sets.
    """

    nodes: tuple[str, ...]
    total_years: float = 0.0
    scores: dict[str, float] = field(default_factory=dict, compare=False, hash=False)

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return " → ".join(self.nodes)


# ---------------------------------------------------------------------------
# CareerGraph
# ---------------------------------------------------------------------------

class CareerGraph:
    """
    Wrapper sobre nx.DiGraph que expone operaciones específicas
    del dominio de trayectorias profesionales.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self._g = graph
        self._validate()

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Verifica que el grafo tenga los atributos mínimos requeridos."""
        required_node_attrs = {"avg_salary", "demand", "satisfaction"}
        required_edge_attrs = {"difficulty", "risk", "transition_years"}

        for node, data in self._g.nodes(data=True):
            missing = required_node_attrs - data.keys()
            if missing:
                raise ValueError(f"Nodo '{node}' falta atributos: {missing}")

        for u, v, data in self._g.edges(data=True):
            missing = required_edge_attrs - data.keys()
            if missing:
                raise ValueError(f"Arista '{u}→{v}' falta atributos: {missing}")

    # ------------------------------------------------------------------
    # Consultas básicas
    # ------------------------------------------------------------------

    def successors(self, node_id: str) -> list[str]:
        """Retorna los nodos alcanzables desde node_id."""
        return list(self._g.successors(node_id))

    def node_attrs(self, node_id: str) -> dict:
        """Retorna todos los atributos de un nodo."""
        return dict(self._g.nodes[node_id])

    def edge_attrs(self, u: str, v: str) -> dict:
        """Retorna atributos de la arista u→v."""
        return dict(self._g[u][v])

    def has_edge(self, u: str, v: str) -> bool:
        return self._g.has_edge(u, v)

    def all_node_ids(self) -> list[str]:
        return list(self._g.nodes())

    # ------------------------------------------------------------------
    # Scoring de trayectorias
    # ------------------------------------------------------------------

    def score_trajectory(self, trajectory: tuple[str, ...]) -> dict[str, float]:
        """
        Calcula métricas cuantitativas de una trayectoria.

        Objetivos (maximizar):
            - salary_growth:  crecimiento salarial promedio entre pasos
            - avg_demand:     demanda laboral promedio de los roles
            - avg_satisfaction: satisfacción promedio de los roles
            - final_salary:   salario del rol final

        Costos (minimizar, devueltos como negativos para uniformidad):
            - total_years:    años totales de la trayectoria
            - avg_risk:       riesgo promedio de las transiciones
            - avg_difficulty: dificultad promedio de las transiciones
        """
        if len(trajectory) < 2:
            return {}

        nodes_data = [self.node_attrs(n) for n in trajectory]
        edges_data = [
            self.edge_attrs(trajectory[i], trajectory[i + 1])
            for i in range(len(trajectory) - 1)
        ]

        salaries = [n["avg_salary"] for n in nodes_data]
        demands = [n["demand"] for n in nodes_data]
        satisfactions = [n["satisfaction"] for n in nodes_data]

        salary_growth = (salaries[-1] - salaries[0]) / max(salaries[0], 1)
        total_years = sum(e["transition_years"] for e in edges_data)
        avg_risk = float(np.mean([e["risk"] for e in edges_data]))
        avg_difficulty = float(np.mean([e["difficulty"] for e in edges_data]))

        return {
            # Objetivos a maximizar
            "salary_growth": round(salary_growth, 4),
            "avg_demand": round(float(np.mean(demands)), 4),
            "avg_satisfaction": round(float(np.mean(satisfactions)), 4),
            "final_salary": round(salaries[-1], 2),
            # Costos (menor es mejor)
            "total_years": round(float(total_years), 2),
            "avg_risk": round(avg_risk, 4),
            "avg_difficulty": round(avg_difficulty, 4),
        }

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def iter_paths_from(
        self,
        source: str,
        max_depth: int = 5,
        visited: frozenset[str] | None = None,
    ) -> Iterator[tuple[str, ...]]:
        """
        Generador: itera todos los caminos simples desde source
        hasta profundidad max_depth. Evita ciclos.
        """
        if visited is None:
            visited = frozenset()

        visited = visited | {source}

        yield (source,)

        if len(visited) >= max_depth:
            return

        for neighbor in self.successors(source):
            if neighbor not in visited:
                for sub_path in self.iter_paths_from(neighbor, max_depth, visited):
                    yield (source,) + sub_path

    def __repr__(self) -> str:
        return (
            f"CareerGraph("
            f"nodes={self._g.number_of_nodes()}, "
            f"edges={self._g.number_of_edges()})"
        )
