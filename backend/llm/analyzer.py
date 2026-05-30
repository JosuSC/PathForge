"""
llm/analyzer.py
---------------
Orquesta el análisis cualitativo de trayectorias usando Gemini.

Responsabilidad: decidir QUÉ analizar, CUÁNDO llamar al LLM,
y CÓMO estructurar la respuesta final que ve el usuario.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from core.evaluator import EvaluatedTrajectory
from llm.client import GeminiClient
from llm.prompts import (
    build_comparison_prompt,
    build_ranking_prompt,
    build_single_analysis_prompt,
)


# ---------------------------------------------------------------------------
# Tipos de resultado
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Resultado del análisis cualitativo de Gemini."""

    analysis_type: str          # "comparison" | "single" | "ranking"
    content: str                # Texto completo de la respuesta
    trajectories_analyzed: int  # Cuántas trayectorias se analizaron
    prompt_used: str            # El prompt enviado (para debugging/informe)

    def __repr__(self) -> str:
        preview = self.content[:100].replace("\n", " ")
        return (
            f"AnalysisResult("
            f"type={self.analysis_type}, "
            f"trajectories={self.trajectories_analyzed}, "
            f"preview='{preview}...')"
        )


# ---------------------------------------------------------------------------
# Analizador principal
# ---------------------------------------------------------------------------

class TrajectoryAnalyzer:
    """
    Usa Gemini para generar análisis cualitativos de trayectorias.

    Tres modos de análisis:
    1. compare()  — compara múltiples trayectorias del frente de Pareto
    2. analyze()  — análisis profundo de una sola trayectoria
    3. rank_by()  — rankea trayectorias según un criterio en lenguaje natural
    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        user_profile: str = "profesional de tecnología",
    ) -> None:
        self._client = client or GeminiClient()
        self._user_profile = user_profile

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def compare(
        self,
        trajectories: list[EvaluatedTrajectory],
        max_trajectories: int = 5,
    ) -> AnalysisResult:
        """
        Compara cualitativamente las mejores trayectorias del frente de Pareto.

        Args:
            trajectories: Lista evaluada (se usan las primeras max_trajectories).
            max_trajectories: Límite para no sobrecargar el contexto del LLM.

        Returns:
            AnalysisResult con el análisis comparativo.
        """
        subset = trajectories[:max_trajectories]
        logger.info(f"Comparando {len(subset)} trayectorias con Gemini...")

        prompt = build_comparison_prompt(subset, self._user_profile)
        response = self._client.complete(prompt)

        return AnalysisResult(
            analysis_type="comparison",
            content=response,
            trajectories_analyzed=len(subset),
            prompt_used=prompt,
        )

    def analyze(self, trajectory: EvaluatedTrajectory) -> AnalysisResult:
        """
        Análisis profundo de una trayectoria específica.

        Args:
            trajectory: Trayectoria a analizar.

        Returns:
            AnalysisResult con el análisis detallado.
        """
        path_str = " → ".join(trajectory.trajectory.nodes)
        logger.info(f"Analizando trayectoria: {path_str}")

        prompt = build_single_analysis_prompt(trajectory, self._user_profile)
        response = self._client.complete(prompt)

        return AnalysisResult(
            analysis_type="single",
            content=response,
            trajectories_analyzed=1,
            prompt_used=prompt,
        )

    def rank_by(
        self,
        trajectories: list[EvaluatedTrajectory],
        criterion: str,
        max_trajectories: int = 5,
    ) -> AnalysisResult:
        """
        Rankea trayectorias según un criterio expresado en lenguaje natural.

        Args:
            trajectories: Lista de trayectorias a rankear.
            criterion: Objetivo del usuario, ej: "quiero ganar mucho pero rápido".
            max_trajectories: Límite de trayectorias a considerar.

        Returns:
            AnalysisResult con el ranking justificado.
        """
        subset = trajectories[:max_trajectories]
        logger.info(f"Rankeando {len(subset)} trayectorias por: '{criterion}'")

        prompt = build_ranking_prompt(subset, criterion)
        response = self._client.complete(prompt)

        return AnalysisResult(
            analysis_type="ranking",
            content=response,
            trajectories_analyzed=len(subset),
            prompt_used=prompt,
        )
