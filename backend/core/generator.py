"""
core/generator.py
-----------------
Genera trayectorias profesionales usando Beam Search multiobjetivo.

Algoritmo: Beam Search + scoring de Pareto
------------------------------------------
1. Comienza desde el nodo origen con un "beam" de K candidatos.
2. En cada paso, expande todos los candidatos por sus vecinos válidos.
3. Filtra por restricciones activas.
4. Puntúa y selecciona los K mejores usando dominancia de Pareto +
   crowding distance (diversidad).
5. Repite hasta alcanzar la profundidad máxima o no haber expansiones.

Fix aplicado: el beam se actualizaba dentro del for interno, causando
sobreescritura prematura. Ahora completed y beam se manejan correctamente.

Optimización: _select_beam usa evaluación directa sin re-instanciar
el evaluador, reduciendo overhead en cada iteración del beam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from core.constraints import Constraint
from core.evaluator import EvaluatedTrajectory, TrajectoryEvaluator
from core.graph import CareerGraph, Trajectory


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """Parámetros del Beam Search."""

    beam_width: int = 10
    """Número de candidatos activos por iteración (K)."""

    max_depth: int = 6
    """Profundidad máxima (número de roles en la trayectoria)."""

    min_depth: int = 2
    """Profundidad mínima para considerar una trayectoria completa."""

    top_k_results: int = 10
    """Número de trayectorias finales a retornar."""

    diversity_weight: float = 0.3
    """Peso de la crowding distance en el score de selección [0, 1]."""

    emit_steps: bool = False
    """Si True, el generador emite cada paso via callback (para WebSocket)."""


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
        self._graph = graph
        self._config = config or GeneratorConfig()
        self._evaluator = TrajectoryEvaluator(graph)

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
            source:          ID del nodo de inicio.
            constraints:     Restricciones a aplicar durante la búsqueda.
            target:          (Opcional) nodo destino deseado.
            step_callback:   Callable(depth, beam, completed) para emitir
                             pasos en tiempo real (WebSocket).

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

        beam: list[tuple[str, ...]] = [(source,)]
        completed: set[tuple[str, ...]] = set()

        for depth in range(1, self._config.max_depth):
            # 1. Expandir beam actual
            candidates = self._expand_beam(beam, constraints, target)

            if not candidates:
                logger.debug(f"Beam vacío en profundidad {depth}.")
                break

            # 2. Registrar trayectorias completadas (min_depth cumplida)
            for path in candidates:
                if len(path) >= self._config.min_depth:
                    completed.add(path)

            # 3. Seleccionar nuevo beam para la siguiente iteración
            beam = self._select_beam(candidates, constraints)

            logger.debug(
                f"Depth {depth}: {len(candidates)} candidatos → "
                f"beam={len(beam)}, completadas={len(completed)}"
            )

            # 4. Emitir paso si hay callback (para animación WebSocket)
            if step_callback and self._config.emit_steps:
                step_callback(depth, list(beam), list(completed))

        # Incluir trayectorias del beam final
        for path in beam:
            if len(path) >= self._config.min_depth:
                completed.add(path)

        logger.info(f"Trayectorias únicas: {len(completed)}")

        # Evaluar y rankear todas
        trajectories = [Trajectory(nodes=p) for p in completed]
        evaluated = self._evaluator.evaluate_all(trajectories)

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

            # Si llegamos al target, no seguimos expandiendo este camino
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
        """
        Selecciona los mejores K candidatos.
        Si hay menos candidatos que beam_width, los retorna todos.
        """
        if len(candidates) <= self._config.beam_width:
            return candidates

        # Evaluar parcialmente para scoring
        partial = [Trajectory(nodes=p) for p in candidates]
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
        crowding = min(et.crowding_distance, 10.0) / 10.0
        return 0.7 * rank_score + 0.3 * crowding

    def _heuristic_potential(self, node_id: str) -> float:
        attrs = self._graph.node_attrs(node_id)
        successors_count = len(self._graph.successors(node_id))
        salary_score = attrs.get("avg_salary", 0) / 180_000
        branching_score = min(successors_count / 5.0, 1.0)
        return 0.6 * salary_score + 0.4 * branching_score