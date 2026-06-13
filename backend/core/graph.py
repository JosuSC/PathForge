"""
core/graph.py
-------------
Capa de abstracción sobre el grafo de carreras.
Pre-computa umbrales por percentil para constraints adaptativos.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterator

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Constante compartida: max años por defecto
# (sincronizada con constraints.py y main_api.py)
# ---------------------------------------------------------------------------
DEFAULT_MAX_YEARS: int = 12


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Trajectory:
    """
    Representa una trayectoria profesional como secuencia de IDs de nodos.
    Inmutable para poder usarla como clave de diccionario y en sets.
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
    Pre-computa umbrales por percentil para que los constraints
    adaptativos funcionen sin recalcular en cada evaluación.
    """

    def __init__(self, graph: nx.DiGraph, outcome_predictor=None) -> None:
        self._g = graph
        self._outcome_predictor = outcome_predictor
        # FIX [GEN3]: calcular max_salary una sola vez para normalización dinámica
        salaries = [
            data.get("avg_salary", 0)
            for _, data in self._g.nodes(data=True)
        ]
        self._max_salary: float = max(salaries) if salaries else 180_000
        self._validate()

        # ── Pre-computar umbrales por percentil para constraints adaptativos ──
        # Se calculan UNA vez al crear el CareerGraph y se reutilizarán
        # en cada llamada a PercentileRiskConstraint.is_satisfied() etc.
        risks = [data.get("risk", 0) for _, _, data in self._g.edges(data=True)]
        diffs = [data.get("difficulty", 0) for _, _, data in self._g.edges(data=True)]

        self._salary_thresholds: dict[str, float] = {}
        if salaries:
            for p in range(1, 100):
                self._salary_thresholds[f"p{p}"] = float(np.percentile(salaries, p))

        self._risk_thresholds: dict[str, float] = {}
        if risks:
            for p in range(1, 100):
                self._risk_thresholds[f"p{p}"] = float(np.percentile(risks, p))

        self._difficulty_thresholds: dict[str, float] = {}
        if diffs:
            for p in range(1, 100):
                self._difficulty_thresholds[f"p{p}"] = float(np.percentile(diffs, p))

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
        return list(self._g.successors(node_id))

    def node_attrs(self, node_id: str) -> dict:
        return dict(self._g.nodes[node_id])

    def edge_attrs(self, u: str, v: str) -> dict:
        return dict(self._g[u][v])

    def has_edge(self, u: str, v: str) -> bool:
        return self._g.has_edge(u, v)

    def all_node_ids(self) -> list[str]:
        return list(self._g.nodes())

    # FIX [G4]: cached_property — se calcula una sola vez
    @cached_property
    def _terminal_nodes_set(self) -> frozenset[str]:
        return frozenset(n for n in self._g.nodes() if self._g.out_degree(n) == 0)

    def terminal_nodes(self) -> list[str]:
        """Retorna nodos terminales (sin aristas salientes). Cacheado."""
        return list(self._terminal_nodes_set)

    def is_terminal(self, node_id: str) -> bool:
        return node_id in self._terminal_nodes_set

    # ------------------------------------------------------------------
    # Scoring de trayectorias
    # ------------------------------------------------------------------

    def score_trajectory(self, trajectory: tuple[str, ...]) -> dict[str, float]:
        """
        Calcula métricas cuantitativas de una trayectoria.

        Objetivos a maximizar:
            salary_growth, avg_demand, avg_satisfaction, final_salary,
            is_terminal_end, transition_probability_score, salary_growth_edge

        Costos (minimizar):
            total_years, avg_risk, avg_difficulty
        """
        if len(trajectory) < 2:
            return {}

        nodes_data  = [self.node_attrs(n) for n in trajectory]
        edges_data  = [
            self.edge_attrs(trajectory[i], trajectory[i + 1])
            for i in range(len(trajectory) - 1)
        ]

        salaries      = [n["avg_salary"]    for n in nodes_data]
        demands       = [n["demand"]        for n in nodes_data]
        satisfactions = [n["satisfaction"]  for n in nodes_data]

        # FIX: abs() protege contra salarios negativos accidentales
        salary_growth = (salaries[-1] - salaries[0]) / max(abs(salaries[0]), 1)
        total_years   = sum(e["transition_years"] for e in edges_data)
        avg_risk       = float(np.mean([e["risk"]       for e in edges_data]))
        avg_difficulty = float(np.mean([e["difficulty"] for e in edges_data]))

        # FIX [G1]: is_terminal_end — usado por main_api para clasificar grupos
        is_terminal_end   = 1.0 if self.is_terminal(trajectory[-1])  else 0.0
        is_terminal_start = 0.0 if self.is_terminal(trajectory[0])   else 1.0

        # FIX [G3]: aprovechar transition_probability y salary_growth de aristas
        trans_probs = [e.get("transition_probability") for e in edges_data]
        trans_probs_valid = [p for p in trans_probs if p is not None]
        transition_probability_score = (
            float(np.mean(trans_probs_valid)) if trans_probs_valid else 0.5
        )

        edge_salary_growths = [e.get("salary_growth") for e in edges_data]
        edge_sal_valid = [g for g in edge_salary_growths if g is not None]
        salary_growth_edge = float(np.mean(edge_sal_valid)) if edge_sal_valid else salary_growth

        return {
            # Objetivos a maximizar
            "salary_growth":               round(salary_growth,               4),
            "avg_demand":                  round(float(np.mean(demands)),      4),
            "avg_satisfaction":            round(float(np.mean(satisfactions)),4),
            "final_salary":                round(salaries[-1],                 2),
            "is_terminal_end":             is_terminal_end,               # FIX [G1]
            "is_terminal_start":           is_terminal_start,
            "transition_probability_score":round(transition_probability_score, 4),  # FIX [G3]
            "salary_growth_edge":          round(salary_growth_edge,          4),   # FIX [G3]
            # Costos (menor es mejor)
            "total_years":    round(float(total_years),   2),
            "avg_risk":       round(avg_risk,             4),
            "avg_difficulty": round(avg_difficulty,       4),
        }

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def iter_paths_from(
        self,
        source: str,
        max_depth: int = 5,
        max_paths: int = 5_000,     # FIX [G2]: límite de paths totales
    ) -> Iterator[tuple[str, ...]]:
        """
        Itera todos los caminos simples desde source hasta max_depth.
        Implementación iterativa con stack explícito.
        Evita recursión profunda y añade límite de paths para grafos densos.
        """
        count = 0
        # Stack: (path_so_far, visited_set)
        stack: list[tuple[tuple[str, ...], frozenset[str]]] = [
            ((source,), frozenset({source}))
        ]

        while stack and count < max_paths:
            path, visited = stack.pop()
            yield path
            count += 1

            if len(path) >= max_depth:
                continue

            current = path[-1]
            for neighbor in self.successors(current):
                if neighbor not in visited:
                    stack.append((path + (neighbor,), visited | {neighbor}))

    def __repr__(self) -> str:
        return (
            f"CareerGraph("
            f"nodes={self._g.number_of_nodes()}, "
            f"edges={self._g.number_of_edges()})"
        )