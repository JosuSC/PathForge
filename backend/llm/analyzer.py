"""
backend/llm/analyzer.py
-----------------------
Orquesta el análisis cualitativo de trayectorias usando el cliente LLM
multi-proveedor con cola circular.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from backend.core.evaluator import EvaluatedTrajectory
from backend.llm.client import LLMClient, get_llm_client
from backend.llm.prompts import (
    build_comparison_prompt,
    build_ranking_prompt,
    build_single_analysis_prompt,
)


@dataclass
class AnalysisResult:
    """Resultado del análisis cualitativo del LLM."""

    analysis_type: str
    content: str
    trajectories_analyzed: int
    prompt_used: str
    provider_used: str = ""

    def __repr__(self) -> str:
        preview = self.content[:100].replace("\n", " ")
        return (
            f"AnalysisResult(type={self.analysis_type}, "
            f"provider={self.provider_used}, "
            f"trajectories={self.trajectories_analyzed}, "
            f"preview='{preview}...')"
        )


class TrajectoryAnalyzer:
    """
    Usa el cliente LLM (multi-proveedor) para análisis cualitativos.

    Tres modos:
        compare()  — compara múltiples trayectorias
        analyze()  — análisis profundo de una trayectoria
        rank_by()  — rankea según criterio en lenguaje natural
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        user_profile: str = "profesional de tecnología",
    ) -> None:
        # Usa el singleton global si no se inyecta uno
        self._client = client or get_llm_client()
        self._user_profile = user_profile

    # ── API pública ──────────────────────────────────────────────

    def compare(
        self,
        trajectories: list[EvaluatedTrajectory],
        max_trajectories: int = 5,
    ) -> AnalysisResult:
        subset = trajectories[:max_trajectories]
        logger.info(f"Comparando {len(subset)} trayectorias...")
        prompt = build_comparison_prompt(subset, self._user_profile)
        response = self._client.complete(prompt)
        return AnalysisResult(
            analysis_type="comparison",
            content=response,
            trajectories_analyzed=len(subset),
            prompt_used=prompt,
            provider_used=self._client.active_provider,
        )

    def analyze(self, trajectory: EvaluatedTrajectory) -> AnalysisResult:
        path_str = " → ".join(trajectory.trajectory.nodes)
        logger.info(f"Analizando: {path_str}")
        prompt = build_single_analysis_prompt(trajectory, self._user_profile)
        response = self._client.complete(prompt)
        return AnalysisResult(
            analysis_type="single",
            content=response,
            trajectories_analyzed=1,
            prompt_used=prompt,
            provider_used=self._client.active_provider,
        )

    def rank_by(
        self,
        trajectories: list[EvaluatedTrajectory],
        criterion: str,
        max_trajectories: int = 5,
    ) -> AnalysisResult:
        subset = trajectories[:max_trajectories]
        logger.info(f"Rankeando {len(subset)} trayectorias por: '{criterion}'")
        prompt = build_ranking_prompt(subset, criterion)
        response = self._client.complete(prompt)
        return AnalysisResult(
            analysis_type="ranking",
            content=response,
            trajectories_analyzed=len(subset),
            prompt_used=prompt,
            provider_used=self._client.active_provider,
        )