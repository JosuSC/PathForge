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

Ventajas sobre BFS/DFS:
- No explota en memoria (solo mantiene K caminos)
- Diversidad garantizada (crowding distance evita convergencia prematura)
- Restricciones aplicadas en tiempo de expansión (poda temprana)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from core.constraints import Constraint
from core.evaluator import EvaluatedTrajectory, TrajectoryEvaluator
from core.graph import CareerGraph, Trajectory


# ---------------------------------------------------------------------------
# Configuración del generador
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """Parámetros del Beam Search."""

    beam_width: int = 10
    """Número de candidatos activos por iteración (K)."""

    max_depth: int = 6
    """Profundidad máxima de la búsqueda (número de roles)."""

    min_depth: int = 2
    """Profundidad mínima para considerar una trayectoria completa."""

    top_k_results: int = 10
    """Número de trayectorias finales a retornar."""

    diversity_weight: float = 0.3
    """Peso de la crowding distance en el score de selección [0, 1]."""


# ---------------------------------------------------------------------------
# Generador principal
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
    ) -> list[EvaluatedTrajectory]:
        """
        Genera y evalúa trayectorias desde `source`.

        Args:
            source: ID del nodo de inicio (rol actual del usuario).
            constraints: Restricciones a aplicar durante la búsqueda.
            target: (Opcional) ID del nodo destino deseado.

        Returns:
            Lista de EvaluatedTrajectory ordenadas por calidad (mejor primero).
        """
        if source not in self._graph.all_node_ids():
            raise ValueError(f"Nodo fuente '{source}' no existe en el grafo.")

        logger.info(
            f"Iniciando Beam Search desde '{source}' | "
            f"beam_width={self._config.beam_width} | "
            f"max_depth={self._config.max_depth}"
        )

        # Beam inicial: una sola trayectoria con el nodo fuente
        beam: list[tuple[str, ...]] = [(source,)]
        completed: list[tuple[str, ...]] = []

        for depth in range(1, self._config.max_depth):
            candidates = self._expand_beam(beam, constraints, target)

            if not candidates:
                logger.debug(f"Beam vacío en profundidad {depth}, deteniendo.")
                break

            # Separar trayectorias completas de las que siguen expandiéndose
            for path in candidates:
                last_node = path[-1]
                is_terminal = (
                    not self._graph.successors(last_node)
                    or (target and last_node == target)
                )
                if len(path) >= self._config.min_depth:
                    completed.append(path)
                if not is_terminal:
                    beam = candidates  # continuar expandiendo

            beam = self._select_beam(candidates, constraints)

            logger.debug(
                f"Depth {depth}: {len(candidates)} candidatos → "
                f"beam={len(beam)}, completadas={len(completed)}"
            )

        # Añadir trayectorias del beam final si son suficientemente largas
        for path in beam:
            if len(path) >= self._config.min_depth:
                completed.append(path)

        # Deduplicar
        unique = list({p: None for p in completed}.keys())
        logger.info(f"Trayectorias únicas generadas: {len(unique)}")

        # Evaluar y rankear
        trajectories = [Trajectory(nodes=p) for p in unique]
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
        Expande cada camino del beam por sus vecinos válidos.
        Aplica restricciones y evita ciclos.
        """
        expanded: list[tuple[str, ...]] = []

        for path in beam:
            current = path[-1]
            visited = set(path)

            for neighbor in self._graph.successors(current):
                if neighbor in visited:
                    continue  # evitar ciclos

                new_path = path + (neighbor,)

                # Poda temprana por restricciones
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
        Selecciona los mejores K candidatos usando:
        1. Score de Pareto (rank + crowding distance)
        2. Heurística de potencial futuro

        Retorna exactamente beam_width candidatos (o menos si no hay suficientes).
        """
        if len(candidates) <= self._config.beam_width:
            return candidates

        # Evaluar candidatos parciales
        partial_trajectories = [Trajectory(nodes=p) for p in candidates]
        evaluated = self._evaluator.evaluate_all(partial_trajectories)

        # Score combinado: calidad de Pareto + potencial heurístico
        scored: list[tuple[float, tuple[str, ...]]] = []
        for et in evaluated:
            quality_score = self._pareto_score(et)
            heuristic_score = self._heuristic_potential(et.trajectory.nodes[-1])
            combined = (
                (1 - self._config.diversity_weight) * quality_score
                + self._config.diversity_weight * heuristic_score
            )
            scored.append((combined, et.trajectory.nodes))

        # Ordenar y retornar los K mejores
        scored.sort(key=lambda x: x[0], reverse=True)
        return [path for _, path in scored[: self._config.beam_width]]

    @staticmethod
    def _pareto_score(et: EvaluatedTrajectory) -> float:
        """
        Convierte rank y crowding distance en un score escalar [0, 1].
        Rank 0 con alta crowding distance = score más alto.
        """
        rank_score = 1.0 / (1.0 + et.pareto_rank)
        crowding = min(et.crowding_distance, 10.0) / 10.0  # normalizar
        return 0.7 * rank_score + 0.3 * crowding

    def _heuristic_potential(self, node_id: str) -> float:
        """
        Heurística: estima el potencial futuro de un nodo.
        Nodos con muchos sucesores y alto salario tienen más potencial.
        """
        attrs = self._graph.node_attrs(node_id)
        successors_count = len(self._graph.successors(node_id))

        # Normalizar salario respecto al máximo posible (~180k en el dataset)
        salary_score = attrs.get("avg_salary", 0) / 180_000
        # Nodos con más opciones futuras son más valiosos
        branching_score = min(successors_count / 5.0, 1.0)

        return 0.6 * salary_score + 0.4 * branching_score
