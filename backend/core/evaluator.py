"""
core/evaluator.py
-----------------
Evalua y rankea trayectorias usando dominancia de Pareto (NSGA-II).

FIX CRITICO: _fast_non_dominated_sort_numpy tenia un bug donde
dominated_mask usaba .any(axis=0) en vez de .sum(axis=0). Esto
causaba que la reduccion del domination_count fuera incorrecta
cuando multiples miembros del frente actual dominaban al mismo
individuo, dejando algunos con rank=-1, lo que provocaba
ZeroDivisionError en _pareto_score (1.0 / (1.0 + (-1)) = div/0).

FIX: Proteccion contra rank=-1 en _pareto_score y _assign_crowding_distances.
FIX: nan_to_num antes y despues de la inversion de costos en _build_objective_matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from backend.core.graph import CareerGraph, Trajectory


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class EvaluatedTrajectory:
    """Trayectoria con sus metricas calculadas y rank de Pareto."""

    trajectory:        Trajectory
    scores:            dict[str, float]
    pareto_rank:       int   = 0
    crowding_distance: float = 0.0

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
    Evalua un conjunto de trayectorias y las ordena por dominancia de Pareto.
    Los objetivos de maximizacion se normalizan en [0, 1].
    Los costos (minimizar) se invierten para uniformidad.
    """

    MAXIMIZE = (
        "salary_growth",
        "avg_demand",
        "avg_satisfaction",
        "final_salary",
        "is_terminal_end",
        "transition_probability_score",
    )
    MINIMIZE = (
        "total_years",
        "avg_risk",
        "avg_difficulty",
    )

    def __init__(self, graph: CareerGraph) -> None:
        self._graph = graph

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def evaluate_all(
        self, trajectories: list[Trajectory]
    ) -> list[EvaluatedTrajectory]:
        """
        Evalua, rankea por Pareto y calcula crowding distance.

        Returns:
            Lista de EvaluatedTrajectory ordenada por rank (mejor primero).
        """
        if not trajectories:
            return []

        valid   = [t for t in trajectories if len(t) >= 2]
        skipped = len(trajectories) - len(valid)
        if skipped:
            logger.debug(f"evaluate_all: {skipped} trayectorias descartadas por len < 2.")

        if not valid:
            return []

        evaluated = [
            EvaluatedTrajectory(
                trajectory=t,
                scores=self._graph.score_trajectory(t.nodes),
            )
            for t in valid
        ]

        obj_matrix = self._build_objective_matrix(evaluated)

        ranks = self._fast_non_dominated_sort_numpy(obj_matrix)
        for et, rank in zip(evaluated, ranks):
            et.pareto_rank = rank

        self._assign_crowding_distances(evaluated, obj_matrix)

        evaluated.sort(key=lambda et: (et.pareto_rank, -et.crowding_distance))
        return evaluated

    def pareto_front(
        self, evaluated: list[EvaluatedTrajectory]
    ) -> list[EvaluatedTrajectory]:
        """Retorna solo las trayectorias del frente optimo (rank == 0)."""
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
            dtype=np.float64,
        )

        # Limpiar NaN/Inf ANTES de invertir costos
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Invertir costos -> todo queda como maximizar
        n_max = len(self.MAXIMIZE)
        matrix[:, n_max:] *= -1

        # Limpiar de nuevo por si la inversion generó valores extremos
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalizar cada columna en [0, 1]
        col_min   = matrix.min(axis=0)
        col_max   = matrix.max(axis=0)
        col_range = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
        matrix    = (matrix - col_min) / col_range

        # Reemplazar NaN por 0.0 (evita divisiones por cero posteriores)
        matrix = np.nan_to_num(matrix, nan=0.0)

        return matrix

    @staticmethod
    def _fast_non_dominated_sort_numpy(obj_matrix: np.ndarray) -> list[int]:
        """
        NSGA-II non-dominated sort vectorizado con numpy.

        FIX CRITICO: La version anterior usaba .any(axis=0) para
        calcular dominated_mask, lo que solo restaba 1 al domination_count
        de cada individuo, incluso si era dominado por multiples miembros
        del frente actual. Esto dejaba algunos individuos con rank=-1,
        causando ZeroDivisionError en _pareto_score.

        Ahora usa .sum(axis=0) para restar el conteo exacto de dominadores.
        """
        n   = len(obj_matrix)
        if n == 0:
            return []

        # obj_matrix shape: (N, M)
        A   = obj_matrix[:, np.newaxis, :]   # shape (N, 1, M)
        B   = obj_matrix[np.newaxis, :, :]   # shape (1, N, M)

        # i domina a j si: para todo m: i[m] >= j[m]  Y  existe m: i[m] > j[m]
        a_ge_b = A >= B      # (N, N, M)
        a_gt_b = A > B       # (N, N, M)

        dominates = np.all(a_ge_b, axis=2) & np.any(a_gt_b, axis=2)  # (N, N)

        # domination_count[i] = cuantos individuos dominan a i
        domination_count = dominates.sum(axis=0).astype(int)

        ranks   = [-1] * n
        current = list(np.where(domination_count == 0)[0])
        rank    = 0

        while current:
            for i in current:
                ranks[i] = rank

            # FIX CRITICO: usar .sum(axis=0) en vez de .any(axis=0)
            # Esto resta el NUMERO EXACTO de dominadores del frente actual,
            # no solo 0 o 1. El bug anterior causaba que nodos dominados
            # por multiples miembros del frente nunca redujeran su count a 0.
            dominated_count = dominates[current, :].sum(axis=0).astype(int)
            domination_count -= dominated_count

            next_front = list(np.where(domination_count == 0)[0])
            # Solo los que aun no tienen rank asignado
            next_front = [i for i in next_front if ranks[i] == -1]
            current = next_front
            rank   += 1

        # FIX DE SEGURIDAD: si algun individuo quedo con rank=-1 (no deberia
        # pasar con el fix de arriba, pero por seguridad), asignarle el rank maximo + 1
        if -1 in ranks:
            max_rank = max(r for r in ranks if r >= 0)
            ranks = [r if r >= 0 else max_rank + 1 for r in ranks]
            logger.warning(
                f"Pareto sort: {sum(1 for r in ranks if r < 0)} individuos "
                f"quedaron sin rank asignado. Se asigno rank={max_rank + 1}."
            )

        return ranks

    @staticmethod
    def _assign_crowding_distances(
        evaluated: list[EvaluatedTrajectory],
        obj_matrix: np.ndarray,
    ) -> None:
        """
        Opera sobre indices locales del frente (no globales).
        Crowding distance mide aislamiento dentro del frente -- mayor = mas diverso.

        FIX: Proteccion contra pareto_rank=-1 y col_range=0.
        """
        # Proteger contra rangos invalidos
        valid_ranks = [et.pareto_rank for et in evaluated if et.pareto_rank >= 0]
        if not valid_ranks:
            for et in evaluated:
                et.crowding_distance = 0.0
            return

        max_rank = max(valid_ranks)

        for rank in range(max_rank + 1):
            global_indices = [i for i, et in enumerate(evaluated) if et.pareto_rank == rank]
            k = len(global_indices)

            if k <= 2:
                for gi in global_indices:
                    evaluated[gi].crowding_distance = float("inf")
                continue

            front_matrix = obj_matrix[global_indices]
            n_obj        = front_matrix.shape[1]
            distances    = np.zeros(k, dtype=np.float64)

            for m in range(n_obj):
                local_sorted = np.argsort(front_matrix[:, m])
                distances[local_sorted[0]]  = float("inf")
                distances[local_sorted[-1]] = float("inf")
                col_range = (
                    front_matrix[local_sorted[-1], m]
                    - front_matrix[local_sorted[0], m]
                )
                if col_range == 0:
                    continue
                for k_idx in range(1, k - 1):
                    distances[local_sorted[k_idx]] += (
                        front_matrix[local_sorted[k_idx + 1], m]
                        - front_matrix[local_sorted[k_idx - 1], m]
                    ) / col_range

            # Limpiar NaN/Inf de las distancias antes de asignar
            distances = np.nan_to_num(distances, nan=0.0, posinf=float("inf"), neginf=0.0)

            for local_i, global_i in enumerate(global_indices):
                evaluated[global_i].crowding_distance = float(distances[local_i])
