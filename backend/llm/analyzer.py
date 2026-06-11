"""
backend/llm/analyzer.py
-----------------------
Orquesta el análisis cualitativo de trayectorias usando el cliente LLM
multi-proveedor con cola circular.

v2 — Fixes:
[AN1] Métodos async añadidos (compare_async, analyze_async, rank_by_async)
      para evitar bloquear el event loop de FastAPI en /api/analyze.
      Los métodos síncronos se mantienen para compatibilidad y uso en
      run_in_executor (WebSocket).
[AN2] Guard clauses en compare(), analyze(), rank_by() cuando trajectories
      está vacío — retorna AnalysisResult informativo sin llamar al LLM.
[AN3] prompt_used eliminado de la respuesta pública del API.
      Se loggea internamente a nivel DEBUG.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from loguru import logger

from backend.core.evaluator import EvaluatedTrajectory
from backend.llm.client import LLMClient, get_llm_client
from backend.llm.prompts import (
    build_comparison_prompt,
    build_ranking_prompt,
    build_single_analysis_prompt,
    build_terminal_comparison_prompt,
)


# ──────────────────────────────────────────────────────────────
# Resultado del análisis
# ──────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """
    Resultado del análisis cualitativo del LLM.

    FIX [AN3]: prompt_used es privado (_prompt_used) — no se serializa
    en respuestas JSON del API. Accesible para logging/debug interno.
    """

    analysis_type:         str
    content:               str
    trajectories_analyzed: int
    provider_used:         str  = ""
    # FIX [AN3]: prefijo _ para señalar que es interno, no se expone en API
    _prompt_used:          str  = field(default="", repr=False)

    def __repr__(self) -> str:
        preview = self.content[:100].replace("\n", " ")
        return (
            f"AnalysisResult(type={self.analysis_type}, "
            f"provider={self.provider_used}, "
            f"trajectories={self.trajectories_analyzed}, "
            f"preview='{preview}...')"
        )

    def to_api_dict(self) -> dict:
        """Serialización segura para respuestas API — sin exponer el prompt."""
        return {
            "analysis_type":         self.analysis_type,
            "content":               self.content,
            "trajectories_analyzed": self.trajectories_analyzed,
            "provider_used":         self.provider_used,
        }


# ──────────────────────────────────────────────────────────────
# Analizador
# ──────────────────────────────────────────────────────────────

class TrajectoryAnalyzer:
    """
    Usa el cliente LLM (multi-proveedor) para análisis cualitativos.

    Cuatro modos:
        compare()       — compara múltiples trayectorias (síncrono)
        analyze()       — análisis profundo de una trayectoria (síncrono)
        rank_by()       — rankea según criterio en lenguaje natural (síncrono)
        compare_terminals() — compara el mejor camino a cada destino final

    Versiones async (*_async) para uso directo en endpoints FastAPI.
    Los síncronos se usan en WebSocket via run_in_executor.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        user_profile: str = "profesional de tecnología",
    ) -> None:
        self._client       = client or get_llm_client()
        self._user_profile = user_profile

    # ── Métodos síncronos (para run_in_executor / WebSocket) ────

    def compare(
        self,
        trajectories: list[EvaluatedTrajectory],
        max_trajectories: int = 5,
    ) -> AnalysisResult:
        # FIX [AN2]: guard clause — no llamar al LLM con lista vacía
        if not trajectories:
            return self._empty_result("comparison")

        subset = trajectories[:max_trajectories]
        logger.info(f"Comparando {len(subset)} trayectorias...")
        prompt   = build_comparison_prompt(subset, self._user_profile)
        logger.debug(f"[compare] prompt ({len(prompt)} chars)")  # FIX [AN3]
        response = self._client.complete(prompt)

        return AnalysisResult(
            analysis_type="comparison",
            content=response,
            trajectories_analyzed=len(subset),
            provider_used=self._client.active_provider,
            _prompt_used=prompt,  # FIX [AN3]: interno, no expuesto en API
        )

    def analyze(self, trajectory: EvaluatedTrajectory) -> AnalysisResult:
        path_str = " → ".join(trajectory.trajectory.nodes)
        logger.info(f"Analizando: {path_str}")
        prompt   = build_single_analysis_prompt(trajectory, self._user_profile)
        logger.debug(f"[analyze] prompt ({len(prompt)} chars)")  # FIX [AN3]
        response = self._client.complete(prompt)

        return AnalysisResult(
            analysis_type="single",
            content=response,
            trajectories_analyzed=1,
            provider_used=self._client.active_provider,
            _prompt_used=prompt,
        )

    def rank_by(
        self,
        trajectories: list[EvaluatedTrajectory],
        criterion: str,
        max_trajectories: int = 5,
    ) -> AnalysisResult:
        # FIX [AN2]: guard clause
        if not trajectories:
            return self._empty_result("ranking")

        subset   = trajectories[:max_trajectories]
        logger.info(f"Rankeando {len(subset)} trayectorias por: '{criterion}'")
        prompt   = build_ranking_prompt(subset, criterion)
        logger.debug(f"[rank_by] prompt ({len(prompt)} chars)")  # FIX [AN3]
        response = self._client.complete(prompt)

        return AnalysisResult(
            analysis_type="ranking",
            content=response,
            trajectories_analyzed=len(subset),
            provider_used=self._client.active_provider,
            _prompt_used=prompt,
        )

    def compare_terminals(
        self,
        terminal_groups: dict[str, list[EvaluatedTrajectory]],
        criterion: str,
    ) -> AnalysisResult:
        """
        FIX [PR4]: método que conecta build_terminal_comparison_prompt
        (que existía en prompts.py pero nunca se usaba).
        Compara el mejor camino hacia cada destino final posible.
        """
        if not terminal_groups:
            return self._empty_result("terminal_comparison")

        logger.info(
            f"Comparando {len(terminal_groups)} destinos finales "
            f"por: '{criterion}'"
        )
        prompt   = build_terminal_comparison_prompt(
            terminal_groups, self._user_profile, criterion
        )
        logger.debug(f"[compare_terminals] prompt ({len(prompt)} chars)")
        response = self._client.complete(prompt)

        total = sum(len(v) for v in terminal_groups.values())
        return AnalysisResult(
            analysis_type="terminal_comparison",
            content=response,
            trajectories_analyzed=total,
            provider_used=self._client.active_provider,
            _prompt_used=prompt,
        )

    # ── Métodos async (FIX [AN1]: para endpoints FastAPI directos) ──

    async def compare_async(
        self,
        trajectories: list[EvaluatedTrajectory],
        max_trajectories: int = 5,
    ) -> AnalysisResult:
        """
        FIX [AN1]: versión async — delega al síncrono en executor para
        no bloquear el event loop de FastAPI. Usable directamente en
        endpoints async sin run_in_executor manual.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.compare, trajectories, max_trajectories
        )

    async def analyze_async(
        self, trajectory: EvaluatedTrajectory
    ) -> AnalysisResult:
        """FIX [AN1]: versión async de analyze()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze, trajectory)

    async def rank_by_async(
        self,
        trajectories: list[EvaluatedTrajectory],
        criterion: str,
        max_trajectories: int = 5,
    ) -> AnalysisResult:
        """FIX [AN1]: versión async de rank_by()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.rank_by, trajectories, criterion, max_trajectories
        )

    async def compare_terminals_async(
        self,
        terminal_groups: dict[str, list[EvaluatedTrajectory]],
        criterion: str,
    ) -> AnalysisResult:
        """FIX [AN1]: versión async de compare_terminals()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.compare_terminals, terminal_groups, criterion
        )

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _empty_result(analysis_type: str) -> AnalysisResult:
        """FIX [AN2]: resultado informativo cuando no hay trayectorias."""
        logger.warning(
            f"TrajectoryAnalyzer.{analysis_type}() llamado con lista vacía. "
            "Retornando resultado vacío sin llamar al LLM."
        )
        return AnalysisResult(
            analysis_type=analysis_type,
            content=(
                "No se encontraron trayectorias para analizar. "
                "Genera primero trayectorias desde un nodo de origen."
            ),
            trajectories_analyzed=0,
            provider_used="none",
        )
