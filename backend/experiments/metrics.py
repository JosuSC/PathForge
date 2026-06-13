"""
experiments/metrics.py
----------------------
Metricas del sistema para el diseno experimental.

FIX ADAPTATIVO: Proteccion contra division por cero en _compute_diversity
y compute_metrics. Cuando pocas trayectorias sobreviven, los rangos
de columnas pueden ser 0, causando float division by zero. Ahora se
reemplazan rangos 0 con 1.0 y se protege con _safe_mean().

FIX NaN/Inf: Se limpian valores NaN/Inf antes de calcular promedios.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from backend.core.evaluator import EvaluatedTrajectory


# ---------------------------------------------------------------------------
# Helper: media segura
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    """
    Calcula la media de forma segura:
    - Lista vacia -> 0.0
    - Filtra NaN e Inf
    - Si todo es NaN/Inf -> 0.0
    """
    if not values:
        return 0.0
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return 0.0
    return float(np.mean(clean))


# ---------------------------------------------------------------------------
# Estructura de metricas
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMetrics:
    """Metricas capturadas durante una ejecucion del generador."""

    config_name:        str
    source_node:        str
    constraint_profile: str

    # Condeos
    trajectories_generated: int   = 0
    pareto_front_size:       int   = 0

    # Calidad promedio (sobre todas las trayectorias generadas)
    avg_salary_growth:  float = 0.0
    avg_satisfaction:   float = 0.0
    avg_risk:           float = 0.0
    avg_years:          float = 0.0

    # FIX [M3]: campos nuevos de graph.py v2
    terminal_rate:          float = 0.0
    avg_transition_prob:    float = 0.0

    # Diversidad
    diversity_score:    float = 0.0

    # Tiempo de ejecucion
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
            "terminal_rate":        round(self.terminal_rate,       4),
            "avg_transition_prob":  round(self.avg_transition_prob, 4),
            "diversity":            round(self.diversity_score,     4),
            "time_ms":              round(self.execution_time_ms,   2),
        }


# ---------------------------------------------------------------------------
# Funcion principal de computo
# ---------------------------------------------------------------------------

def compute_metrics(
    results:            list[EvaluatedTrajectory],
    config_name:        str,
    source_node:        str,
    constraint_profile: str,
    execution_time_ms:  float,
) -> ExperimentMetrics:
    """
    Calcula todas las metricas para una ejecucion del experimento.
    """
    if not results:
        return ExperimentMetrics(
            config_name=config_name,
            source_node=source_node,
            constraint_profile=constraint_profile,
            execution_time_ms=execution_time_ms,
        )

    scores_list = [et.scores for et in results]

    pareto_size = sum(1 for et in results if et.pareto_rank == 0)

    avg_salary_growth = _safe_mean([s.get("salary_growth",      0.0) for s in scores_list])
    avg_satisfaction  = _safe_mean([s.get("avg_satisfaction",   0.0) for s in scores_list])
    avg_risk          = _safe_mean([s.get("avg_risk",           0.0) for s in scores_list])
    avg_years         = _safe_mean([s.get("total_years",        0.0) for s in scores_list])

    terminal_vals  = [s.get("is_terminal_end", 0.0) for s in scores_list]
    terminal_rate  = _safe_mean(terminal_vals)

    trans_vals = [s.get("transition_probability_score") for s in scores_list]
    trans_valid = [v for v in trans_vals if v is not None and math.isfinite(v)]
    avg_transition_prob = float(np.mean(trans_valid)) if trans_valid else 0.0

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
# Diversidad vectorizada -- con proteccion contra division por cero
# ---------------------------------------------------------------------------

def _compute_diversity(results: list[EvaluatedTrajectory]) -> float:
    """
    Diversidad = distancia euclidiana media entre todos los pares
    de trayectorias en el espacio de scores normalizados.

    Vectorizado con numpy broadcasting.

    FIX ADAPTATIVO: Cuando todas las trayectorias tienen el mismo valor
    en una columna (col_range=0), reemplazamos el rango por 1.0.
    Tambien protegemos contra NaN/Inf en los resultados.
    """
    if len(results) < 2:
        return 0.0

    keys = [
        "salary_growth", "avg_demand", "avg_satisfaction",
        "total_years", "avg_risk", "avg_difficulty",
    ]

    # Construir matriz, reemplazando NaN/Inf con 0.0
    raw_matrix = []
    for et in results:
        row = []
        for k in keys:
            v = et.scores.get(k, 0.0)
            if not math.isfinite(v):
                v = 0.0
            row.append(v)
        raw_matrix.append(row)

    matrix = np.array(raw_matrix, dtype=np.float64)

    # Normalizar columnas en [0, 1]
    col_min   = matrix.min(axis=0)
    col_max   = matrix.max(axis=0)
    col_range = col_max - col_min
    col_range = np.where(col_range == 0, 1.0, col_range)
    matrix    = (matrix - col_min) / col_range

    # Limpiar NaN/Inf resultantes de la normalizacion
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)

    # Broadcasting matricial
    diff = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    # Extraer triangulo superior (sin diagonal) para pares unicos
    n          = len(results)
    upper_idx  = np.triu_indices(n, k=1)
    pairwise   = dist[upper_idx]

    # Filtrar NaN/Inf antes de calcular la media
    pairwise_clean = pairwise[np.isfinite(pairwise)]

    return float(pairwise_clean.mean()) if len(pairwise_clean) > 0 else 0.0
