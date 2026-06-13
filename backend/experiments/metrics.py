"""
experiments/metrics.py
----------------------
Métricas del sistema para el diseño experimental.

Mide el COMPORTAMIENTO del algoritmo, no solo la calidad de las
trayectorias. Permite comparar configuraciones científicamente:
    - ¿Qué beam_width produce más diversidad?
    - ¿Qué perfil de restricciones genera más trayectorias Pareto-óptimas?
    - ¿Cuánto tiempo tarda cada configuración?

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from backend.core.evaluator import EvaluatedTrajectory  # FIX [M1]


# ---------------------------------------------------------------------------
# Estructura de métricas
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMetrics:
    """Métricas capturadas durante una ejecución del generador."""

    config_name:        str
    source_node:        str
    constraint_profile: str

    # Conteos
    trajectories_generated: int   = 0
    pareto_front_size:       int   = 0

    # Calidad promedio (sobre todas las trayectorias generadas)
    avg_salary_growth:  float = 0.0
    avg_satisfaction:   float = 0.0
    avg_risk:           float = 0.0
    avg_years:          float = 0.0

    # FIX [M3]: campos nuevos de graph.py v2
    terminal_rate:          float = 0.0   # fracción de trayectorias que terminan en nodo terminal
    avg_transition_prob:    float = 0.0   # probabilidad media de transición (dato real, no sintético)

    # Diversidad (cuán distintas son las trayectorias entre sí)
    diversity_score:    float = 0.0

    # Tiempo de ejecución
    execution_time_ms:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "config":               self.config_name,
            "source":               self.source_node,
            "profile":              self.constraint_profile,
            "n_trajectories":       self.trajectories_generated,
            "pareto_front_size":    self.pareto_front_size,
            "avg_salary_growth":    round(self.avg_salary_growth,   4),
            "avg_satisfaction":     round(self.avg_satisfaction,    4),
            "avg_risk":             round(self.avg_risk,            4),
            "avg_years":            round(self.avg_years,           2),
            # FIX [M3]: incluidos en la salida
            "terminal_rate":        round(self.terminal_rate,       4),
            "avg_transition_prob":  round(self.avg_transition_prob, 4),
            "diversity":            round(self.diversity_score,     4),
            "time_ms":              round(self.execution_time_ms,   2),
        }


# ---------------------------------------------------------------------------
# Función principal de cómputo
# ---------------------------------------------------------------------------

def compute_metrics(
    results:            list[EvaluatedTrajectory],
    config_name:        str,
    source_node:        str,
    constraint_profile: str,
    execution_time_ms:  float,
) -> ExperimentMetrics:
    """
    Calcula todas las métricas para una ejecución del experimento.

    La diversidad se mide como la distancia euclidiana media entre todos
    los pares de trayectorias en el espacio de scores normalizados.
    """
    if not results:
        return ExperimentMetrics(
            config_name=config_name,
            source_node=source_node,
            constraint_profile=constraint_profile,
            execution_time_ms=execution_time_ms,
        )

    scores_list = [et.scores for et in results]

    # FIX [M4]: lista comprehension limpia
    pareto_size = sum(1 for et in results if et.pareto_rank == 0)

    def _safe_mean(values: list) -> float:
        """Media segura que devuelve 0.0 si la lista está vacía o tiene NaN/Inf."""
        if not values:
            return 0.0
        arr = np.array(values, dtype=np.float64)
        result = float(np.nanmean(arr))
        return result if not (np.isnan(result) or np.isinf(result)) else 0.0

    avg_salary_growth = _safe_mean([s.get("salary_growth",      0.0) for s in scores_list])
    avg_satisfaction  = _safe_mean([s.get("avg_satisfaction",   0.0) for s in scores_list])
    avg_risk          = _safe_mean([s.get("avg_risk",           0.0) for s in scores_list])
    avg_years         = _safe_mean([s.get("total_years",        0.0) for s in scores_list])

    # FIX [M3]: métricas nuevas de graph.py v2
    terminal_vals  = [s.get("is_terminal_end", 0.0) for s in scores_list]
    terminal_rate  = _safe_mean(terminal_vals)

    trans_vals = [s.get("transition_probability_score") for s in scores_list]
    trans_valid = [v for v in trans_vals if v is not None]
    avg_transition_prob = _safe_mean(trans_valid) if trans_valid else 0.0

    diversity = _compute_diversity(results)

    return ExperimentMetrics(
        config_name=config_name,
        source_node=source_node,
        constraint_profile=constraint_profile,
        trajectories_generated=len(results),
        pareto_front_size=pareto_size,
        avg_salary_growth=avg_salary_growth,
        avg_satisfaction=avg_satisfaction,
        avg_risk=avg_risk,
        avg_years=avg_years,
        terminal_rate=terminal_rate,
        avg_transition_prob=avg_transition_prob,
        diversity_score=diversity,
        execution_time_ms=execution_time_ms,
    )


# ---------------------------------------------------------------------------
# Diversidad vectorizada
# ---------------------------------------------------------------------------

def _compute_diversity(results: list[EvaluatedTrajectory]) -> float:
    """
    Diversidad = distancia euclidiana media entre todos los pares
    de trayectorias en el espacio de scores normalizados.

    Vectorizado con numpy broadcasting — O(N) operaciones numpy
    en lugar de O(N²) loops Python. Para N=100: ~100x más rápido.
    
    """
    if len(results) < 2:
        return 0.0

    keys = [
        "salary_growth", "avg_demand", "avg_satisfaction",
        "total_years", "avg_risk", "avg_difficulty",
    ]

    matrix = np.array(
        [[et.scores.get(k, 0.0) for k in keys] for et in results],
        dtype=np.float64,
    )

    # Normalizar columnas en [0, 1]
    col_min   = matrix.min(axis=0)
    col_max   = matrix.max(axis=0)
    col_range = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    matrix    = (matrix - col_min) / col_range

    # FIX [M2]: broadcasting matricial — sin loops Python
    # diff[i, j, :] = matrix[i, :] - matrix[j, :]  shape: (N, N, M)
    diff = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]    # (N, N, M)
    dist = np.sqrt((diff ** 2).sum(axis=2))                        # (N, N)

    # Extraer triángulo superior (sin diagonal) para pares únicos
    n          = len(results)
    upper_idx  = np.triu_indices(n, k=1)
    pairwise   = dist[upper_idx]

    mean_val = float(pairwise.mean()) if len(pairwise) > 0 else 0.0
    # Proteger contra NaN/Inf
    if np.isnan(mean_val) or np.isinf(mean_val):
        return 0.0
    return mean_val
