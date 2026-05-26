"""
core/evaluator.py
-----------------
Evalúa y rankea trayectorias usando dominancia de Pareto.

Concepto clave — Dominancia de Pareto:
    Una trayectoria A *domina* a B si A es al menos igual en todos los
    objetivos y estrictamente mejor en al menos uno.
    El *frente de Pareto* es el conjunto de trayectorias no dominadas:
    las que representan los mejores trade-offs posibles.

Esto es superior a un ranking simple porque no obliga al usuario a
elegir un único criterio "ganador" — le muestra el espacio de trade-offs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.graph import CareerGraph, Trajectory


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class EvaluatedTrajectory:
    """Trayectoria con sus métricas calculadas y rank de Pareto."""

    trajectory: Trajectory
    scores: dict[str, float]
    pareto_rank: int = 0          # 0 = frente óptimo, mayor = peor
    crowding_distance: float = 0.0  # Diversidad dentro del frente

    def __repr__(self) -> str:
        return (
            f"EvaluatedTrajectory("
            f"path={self.trajectory}, "
            f"rank={self.pareto_rank}, "
            f"salary_growth={self.scores.get('salary_growth', 0):.2f})"
        )


# ---------------------------------------------------------------------------
# Evaluador principal
# ---------------------------------------------------------------------------

class TrajectoryEvaluator:
    """
    Evalúa un conjunto de trayectorias y las ordena por dominancia de Pareto.

    Los objetivos de maximización se normalizan en [0, 1].
    Los costos (minimizar) se invierten para uniformidad.
    """

    # Objetivos a maximizar (mayor = mejor)
    MAXIMIZE = ("salary_growth", "avg_demand", "avg_satisfaction", "final_salary")
    # Costos a minimizar (menor = mejor → se invierten)
    MINIMIZE = ("total_years", "avg_risk", "avg_difficulty")

    def __init__(self, graph: CareerGraph) -> None:
        self._graph = graph

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def evaluate_all(
        self, trajectories: list[Trajectory]
    ) -> list[EvaluatedTrajectory]:
        """
        Evalúa, rankea por Pareto y calcula crowding distance.

        Args:
            trajectories: Lista de trayectorias a evaluar.

        Returns:
            Lista de EvaluatedTrajectory ordenada por rank (mejor primero).
        """
        if not trajectories:
            return []

        # 1. Calcular scores brutos
        evaluated = [
            EvaluatedTrajectory(
                trajectory=t,
                scores=self._graph.score_trajectory(t.nodes),
            )
            for t in trajectories
            if len(t) >= 2
        ]

        if not evaluated:
            return []

        # 2. Construir matriz de objetivos normalizados
        obj_matrix = self._build_objective_matrix(evaluated)

        # 3. Asignar ranks de Pareto (NSGA-II fast non-dominated sort)
        ranks = self._fast_non_dominated_sort(obj_matrix)
        for et, rank in zip(evaluated, ranks):
            et.pareto_rank = rank

        # 4. Calcular crowding distance por frente
        self._assign_crowding_distances(evaluated, obj_matrix)

        # 5. Ordenar: menor rank primero, mayor crowding distance desempata
        evaluated.sort(key=lambda et: (et.pareto_rank, -et.crowding_distance))

        return evaluated

    def pareto_front(
        self, evaluated: list[EvaluatedTrajectory]
    ) -> list[EvaluatedTrajectory]:
        """Retorna solo las trayectorias del frente óptimo (rank == 0)."""
        return [et for et in evaluated if et.pareto_rank == 0]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_objective_matrix(
        self, evaluated: list[EvaluatedTrajectory]
    ) -> np.ndarray:
        """
        Construye matriz (N x M) donde cada fila es una trayectoria
        y cada columna un objetivo, todos orientados a MAXIMIZAR.
        """
        objectives = list(self.MAXIMIZE) + list(self.MINIMIZE)
        matrix = np.array(
            [[et.scores.get(obj, 0.0) for obj in objectives] for et in evaluated],
            dtype=float,
        )

        # Invertir columnas de minimización → todo queda como maximizar
        n_max = len(self.MAXIMIZE)
        matrix[:, n_max:] *= -1

        # Normalizar cada columna en [0, 1]
        col_min = matrix.min(axis=0)
        col_max = matrix.max(axis=0)
        col_range = np.where(col_max - col_min == 0, 1, col_max - col_min)
        matrix = (matrix - col_min) / col_range

        return matrix

    @staticmethod
    def _fast_non_dominated_sort(obj_matrix: np.ndarray) -> list[int]:
        """
        NSGA-II fast non-dominated sort.
        Retorna lista de ranks (0-indexed) para cada individuo.

        Complejidad: O(M * N²) donde M = objetivos, N = individuos.
        """
        n = len(obj_matrix)
        domination_count = np.zeros(n, dtype=int)  # cuántos dominan a i
        dominated_by: list[list[int]] = [[] for _ in range(n)]
        ranks = [-1] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if TrajectoryEvaluator._dominates(obj_matrix[i], obj_matrix[j]):
                    dominated_by[i].append(j)
                elif TrajectoryEvaluator._dominates(obj_matrix[j], obj_matrix[i]):
                    domination_count[i] += 1

        # Frente 0: nadie los domina
        current_front = [i for i in range(n) if domination_count[i] == 0]
        rank = 0
        while current_front:
            for i in current_front:
                ranks[i] = rank
            next_front = []
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
            rank += 1

        return ranks

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """
        a domina a b si:
        - a >= b en todos los objetivos, Y
        - a > b en al menos uno.
        """
        return bool(np.all(a >= b) and np.any(a > b))

    @staticmethod
    def _assign_crowding_distances(
        evaluated: list[EvaluatedTrajectory],
        obj_matrix: np.ndarray,
    ) -> None:
        """
        Crowding distance: mide qué tan "aislada" está cada solución
        dentro de su frente. Mayor distancia = más diversidad = mejor.
        Permite al algoritmo preferir soluciones diversas sobre clusters.
        """
        max_rank = max(et.pareto_rank for et in evaluated)

        for rank in range(max_rank + 1):
            indices = [i for i, et in enumerate(evaluated) if et.pareto_rank == rank]
            if len(indices) <= 2:
                for i in indices:
                    evaluated[i].crowding_distance = float("inf")
                continue

            front_matrix = obj_matrix[indices]
            n_obj = front_matrix.shape[1]
            distances = np.zeros(len(indices))

            for m in range(n_obj):
                sorted_idx = np.argsort(front_matrix[:, m])
                distances[sorted_idx[0]] = float("inf")
                distances[sorted_idx[-1]] = float("inf")
                col_range = front_matrix[sorted_idx[-1], m] - front_matrix[sorted_idx[0], m]
                if col_range == 0:
                    continue
                for k in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[k]] += (
                        front_matrix[sorted_idx[k + 1], m]
                        - front_matrix[sorted_idx[k - 1], m]
                    ) / col_range

            for local_i, global_i in enumerate(indices):
                evaluated[global_i].crowding_distance = float(distances[local_i])
