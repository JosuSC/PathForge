"""
experiments/metrics.py
----------------------
Métricas del sistema para el diseño experimental.

Mide el COMPORTAMIENTO del algoritmo, no solo la calidad de las
trayectorias. Esto permite comparar configuraciones científicamente:
    - ¿Qué beam_width produce más diversidad?
    - ¿Qué perfil de restricciones genera más trayectorias Pareto-óptimas?
    - ¿Cuánto tiempo tarda cada configuración?
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from core.evaluator import EvaluatedTrajectory


@dataclass
class ExperimentMetrics:
    """Métricas capturadas durante una ejecución del generador."""

    config_name: str
    source_node: str
    constraint_profile: str

    # Conteos
    trajectories_generated: int = 0
    pareto_front_size: int = 0

    # Calidad del frente de Pareto
    avg_salary_growth: float = 0.0
    avg_satisfaction: float = 0.0
    avg_risk: float = 0.0
    avg_years: float = 0.0

    # Diversidad (cuán distintas son las trayectorias entre sí)
    diversity_score: float = 0.0

    # Tiempo de ejecución
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "config": self.config_name,
            "source": self.source_node,
            "profile": self.constraint_profile,
            "n_trajectories": self.trajectories_generated,
            "pareto_front_size": self.pareto_front_size,
            "avg_salary_growth": round(self.avg_salary_growth, 4),
            "avg_satisfaction": round(self.avg_satisfaction, 4),
            "avg_risk": round(self.avg_risk, 4),
            "avg_years": round(self.avg_years, 2),
            "diversity": round(self.diversity_score, 4),
            "time_ms": round(self.execution_time_ms, 2),
        }


def compute_metrics(
    results: list[EvaluatedTrajectory],
    config_name: str,
    source_node: str,
    constraint_profile: str,
    execution_time_ms: float,
) -> ExperimentMetrics:
    """
    Calcula todas las métricas para una ejecución del experimento.

    La diversidad se mide como la distancia media entre pares de
    trayectorias en el espacio de scores normalizados.
    """
    if not results:
        return ExperimentMetrics(
            config_name=config_name,
            source_node=source_node,
            constraint_profile=constraint_profile,
            execution_time_ms=execution_time_ms,
        )

    pareto = [et for et in results if et.pareto_rank == 0]

    scores_list = [et.scores for et in results]

    avg_salary_growth = float(np.mean([s.get("salary_growth", 0) for s in scores_list]))
    avg_satisfaction  = float(np.mean([s.get("avg_satisfaction", 0) for s in scores_list]))
    avg_risk          = float(np.mean([s.get("avg_risk", 0) for s in scores_list]))
    avg_years         = float(np.mean([s.get("total_years", 0) for s in scores_list]))

    diversity = _compute_diversity(results)

    return ExperimentMetrics(
        config_name=config_name,
        source_node=source_node,
        constraint_profile=constraint_profile,
        trajectories_generated=len(results),
        pareto_front_size=len(pareto),
        avg_salary_growth=avg_salary_growth,
        avg_satisfaction=avg_satisfaction,
        avg_risk=avg_risk,
        avg_years=avg_years,
        diversity_score=diversity,
        execution_time_ms=execution_time_ms,
    )


def _compute_diversity(results: list[EvaluatedTrajectory]) -> float:
    """
    Diversidad = distancia euclidiana media entre todos los pares
    de trayectorias en el espacio de scores normalizados.

    Mayor diversidad → el algoritmo explora mejor el espacio.
    """
    if len(results) < 2:
        return 0.0

    keys = ["salary_growth", "avg_demand", "avg_satisfaction",
            "total_years", "avg_risk", "avg_difficulty"]

    matrix = np.array(
        [[et.scores.get(k, 0.0) for k in keys] for et in results],
        dtype=float,
    )

    # Normalizar columnas
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_range = np.where(col_max - col_min == 0, 1, col_max - col_min)
    matrix = (matrix - col_min) / col_range

    # Distancia media entre todos los pares
    n = len(matrix)
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += float(np.linalg.norm(matrix[i] - matrix[j]))
            count += 1

    return total_dist / count if count > 0 else 0.0
