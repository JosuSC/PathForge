"""
core/generator.py
-----------------
Genera trayectorias profesionales usando Beam Search multiobjetivo.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from backend.core.constraints import Constraint
from backend.core.evaluator import EvaluatedTrajectory, TrajectoryEvaluator
from backend.core.graph import CareerGraph, Trajectory


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """Parámetros del Beam Search."""

    beam_width:      int   = 10
    max_depth:       int   = 6
    min_depth:       int   = 2
    top_k_results:   int   = 10
    diversity_weight: float = 0.3
    emit_steps:      bool  = False


# ---------------------------------------------------------------------------
# Generador
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    """
    Genera trayectorias profesionales diversas y de alta calidad
    usando Beam Search guiado por dominancia de Pareto.
    """

    def __init__(
        self,
        graph: CareerGraph,
        config: GeneratorConfig | None = None,
    ) -> None:
        # Validar que el grafo sea válido ANTES de leer sus atributos
        if not hasattr(graph, '_max_salary'):
            raise TypeError("El grafo debe implementar los métodos y atributos de CareerGraph")
        
        self._graph     = graph
        self._config    = config or GeneratorConfig()
        self._evaluator = TrajectoryEvaluator(graph)
        # Max_salary dinámico, no hardcoded a 180_000
        self._max_salary: float = graph._max_salary or 180_000

        # Verificar que el grafo tiene la interfaz que Constraint necesita
        for method in ("node_attrs", "edge_attrs", "has_edge"):
            if not callable(getattr(graph, method, None)):
                raise TypeError(
                    f"CareerGraph debe implementar '{method}' para ser usado con Constraints."
                )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def generate(
        self,
        source: str,
        constraints: Constraint | None = None,
        target: str | None = None,
        step_callback=None,
    ) -> list[EvaluatedTrajectory]:
        """
        Genera y evalúa trayectorias desde `source`.

        Args:
            source:         ID del nodo de inicio.
            constraints:    Restricciones a aplicar (branch-and-bound).
            target:         Nodo destino opcional.
            step_callback:  Callable(depth, expanded, selected_beam, completed)
                            para emitir pasos en tiempo real (WebSocket).
                            FIX [GEN4]: recibe tanto expandidos como seleccionados.

        Returns:
            Lista de EvaluatedTrajectory ordenada por calidad (mejor primero).
        """
        if source not in self._graph.all_node_ids():
            raise ValueError(f"Nodo fuente '{source}' no existe en el grafo.")

        logger.info(
            f"Beam Search desde '{source}' | "
            f"beam_width={self._config.beam_width} | "
            f"max_depth={self._config.max_depth}"
        )

        beam:      list[tuple[str, ...]] = [(source,)]
        completed: set[tuple[str, ...]]  = set()

        for depth in range(1, self._config.max_depth):
            candidates = self._expand_beam(beam, constraints, target)

            if not candidates:
                logger.debug(f"Beam vacío en profundidad {depth}.")
                break

            # Deduplicar antes de evaluar y seleccionar
            candidates = list(dict.fromkeys(candidates))

            # Registrar trayectorias completadas
            for path in candidates:
                if len(path) >= self._config.min_depth:
                    completed.add(path)

            # Seleccionar nuevo beam
            new_beam = self._select_beam(candidates, constraints)

            logger.debug(
                f"Depth {depth}: {len(candidates)} candidatos → "
                f"beam={len(new_beam)}, completadas={len(completed)}"
            )

            # Callback recibe expanded Y selected para visualización completa
            if step_callback and self._config.emit_steps:
                step_callback(depth, candidates, list(new_beam), list(completed))

            beam = new_beam

        # Incluir trayectorias del beam final
        for path in beam:
            if len(path) >= self._config.min_depth:
                completed.add(path)

        logger.info(f"Trayectorias únicas: {len(completed)}")

        trajectories = [Trajectory(nodes=p) for p in completed]
        evaluated    = self._evaluator.evaluate_all(trajectories)
        return evaluated[: self._config.top_k_results]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _expand_beam(
        self,
        beam: list[tuple[str, ...]],
        constraints: Constraint | None,
        target: str | None,
    ) -> list[tuple[str, ...]]:
        """
        Expande cada camino del beam. Evita ciclos y aplica poda por
        restricciones en tiempo de expansión (branch-and-bound).
        """
        expanded: list[tuple[str, ...]] = []

        for path in beam:
            current = path[-1]
            visited = set(path)

            if target and current == target:
                continue

            for neighbor in self._graph.successors(current):
                if neighbor in visited:
                    continue
                new_path = path + (neighbor,)
                if constraints and not constraints.is_satisfied(new_path, self._graph):
                    continue
                expanded.append(new_path)

        return expanded

    def _select_beam(
        self,
        candidates: list[tuple[str, ...]],
        constraints: Constraint | None,
    ) -> list[tuple[str, ...]]:
        """Selecciona los mejores K candidatos por Pareto + heurística."""
        if len(candidates) <= self._config.beam_width:
            return candidates

        partial  = [Trajectory(nodes=p) for p in candidates]
        evaluated = self._evaluator.evaluate_all(partial)

        scored: list[tuple[float, tuple[str, ...]]] = [
            (
                (1 - self._config.diversity_weight) * self._pareto_score(et)
                + self._config.diversity_weight * self._heuristic_potential(et.trajectory.nodes[-1]),
                et.trajectory.nodes,
            )
            for et in evaluated
        ]

        scored.sort(key=lambda x: x[0], reverse=True)
        return [path for _, path in scored[: self._config.beam_width]]

    @staticmethod
    def _pareto_score(et: EvaluatedTrajectory) -> float:
        rank_score = 1.0 / (1.0 + et.pareto_rank)
        crowding   = min(et.crowding_distance, 10.0) / 10.0
        return 0.7 * rank_score + 0.3 * crowding

    def _heuristic_potential(self, node_id: str) -> float:
        """
        Normalización dinámica con max_salary del grafo actual,
        no hardcoded a 180_000. Evita saturación en grafos con salarios altos.
        """
        attrs            = self._graph.node_attrs(node_id)
        successors_count = len(self._graph.successors(node_id))
        # Self._max_salary calculado dinámicamente en __init__
        salary_score     = attrs.get("avg_salary", 0) / max(self._max_salary, 1)
        branching_score  = min(successors_count / 5.0, 1.0)
        return 0.6 * salary_score + 0.4 * branching_score
