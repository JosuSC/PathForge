"""
tests/test_llm.py
-----------------
Tests del sistema LLM multi-proveedor.

Cubre:
    - Cliente: guard sin API keys, singleton, reset
    - Analyzer: guard con lista vacía [FIX AN2]
    - Analyzer: to_api_dict no expone prompt [FIX AN3]
    - Analyzer: métodos async presentes [FIX AN1]
    - Prompts: ml_success_prob eliminado [FIX PR1]
    - Prompts: sim_section condicional [FIX PR2]
    - Prompts: terminal_groups precalculado [FIX PR3]
    - Prompts: build_terminal_comparison_prompt usable [FIX PR4]
    - Prompts: etiquetas en inglés [FIX PR5]
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.graph import Trajectory
from backend.core.evaluator import EvaluatedTrajectory


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

def _make_traj(nodes: tuple, salary_growth=1.0, final_salary=90_000,
               avg_demand=0.8, avg_satisfaction=0.75, total_years=5.0,
               avg_risk=0.3, avg_difficulty=0.4, transition_prob=0.45,
               is_terminal_end=1.0, pareto_rank=0) -> EvaluatedTrajectory:
    return EvaluatedTrajectory(
        trajectory=Trajectory(nodes=nodes),
        scores={
            "salary_growth":               salary_growth,
            "final_salary":                final_salary,
            "avg_demand":                  avg_demand,
            "avg_satisfaction":            avg_satisfaction,
            "total_years":                 total_years,
            "avg_risk":                    avg_risk,
            "avg_difficulty":              avg_difficulty,
            "transition_probability_score":transition_prob,
            "is_terminal_end":             is_terminal_end,
        },
        pareto_rank=pareto_rank,
    )


@pytest.fixture
def sample_trajectories() -> list[EvaluatedTrajectory]:
    return [
        _make_traj(("junior_dev", "mid_dev", "cto"), salary_growth=6.2, final_salary=180_000),
        _make_traj(("junior_dev", "data_scientist", "ml_engineer"), salary_growth=2.4, final_salary=85_000, pareto_rank=1),
        _make_traj(("junior_dev", "devops_engineer", "tech_lead"), salary_growth=2.6, final_salary=90_000, pareto_rank=0),
    ]


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.active_provider = "gemini"
    client.complete.return_value = "Análisis mock: La trayectoria A es superior por su mayor crecimiento salarial."
    return client


# ──────────────────────────────────────────────────────────────
# Tests del cliente LLM
# ──────────────────────────────────────────────────────────────

class TestLLMClient:

    def test_client_raises_without_keys(self):
        """[FIX CL6] Sin API keys configuradas lanza EnvironmentError."""
        from backend.llm.client import LLMClient, reset_llm_client
        reset_llm_client()
        with patch.dict("os.environ", {}, clear=True):
            # Limpiar todas las LLM_KEY_N
            env_without_keys = {
                k: v for k, v in __import__("os").environ.items()
                if not k.startswith("LLM_KEY_") and k != "GEMINI_API_KEY"
            }
            with patch.dict("os.environ", env_without_keys, clear=True):
                with pytest.raises(EnvironmentError, match="API keys"):
                    LLMClient()
        reset_llm_client()

    def test_reset_clears_singleton(self):
        """[FIX CL6] reset_llm_client() permite recrear el singleton."""
        from backend.llm.client import reset_llm_client, _client_instance
        reset_llm_client()
        # Después del reset, _client_instance debe ser None
        import backend.llm.client as c_mod
        assert c_mod._client_instance is None
        assert c_mod._client_error is None

    def test_all_providers_have_callers(self):
        """[FIX CL1] Todos los proveedores del Enum tienen caller registrado."""
        from backend.llm.client import Provider, _PROVIDER_CALLERS, _DEFAULT_MODELS
        for provider in Provider:
            assert provider in _PROVIDER_CALLERS, \
                f"Provider.{provider.name} sin caller en _PROVIDER_CALLERS"
            assert provider in _DEFAULT_MODELS, \
                f"Provider.{provider.name} sin modelo en _DEFAULT_MODELS"

    def test_groq_deepseek_mistral_registered(self):
        """[FIX CL1] Los 3 nuevos proveedores están registrados."""
        from backend.llm.client import Provider, _PROVIDER_CALLERS
        for p_name in ("groq", "deepseek", "mistral"):
            provider = Provider(p_name)
            assert provider in _PROVIDER_CALLERS, f"{p_name} no tiene caller"

    def test_quota_error_detection(self):
        """_is_quota_error detecta los patrones correctos."""
        from backend.llm.client import _is_quota_error

        quota_errors = [
            Exception("quota exceeded"),
            Exception("rate limit reached"),
            Exception("HTTP 429"),
            Exception("RESOURCE_EXHAUSTED"),
            Exception("overloaded"),
        ]
        non_quota_errors = [
            Exception("connection timeout"),
            Exception("invalid JSON"),
            Exception("SSL error"),
        ]

        for exc in quota_errors:
            assert _is_quota_error(exc), f"Debería detectar quota: {exc}"
        for exc in non_quota_errors:
            assert not _is_quota_error(exc), f"No debería ser quota: {exc}"


# ──────────────────────────────────────────────────────────────
# Tests del Analyzer
# ──────────────────────────────────────────────────────────────

class TestTrajectoryAnalyzer:

    def test_compare_empty_list_no_llm_call(self, mock_client):
        """[FIX AN2] compare([]) retorna resultado vacío sin llamar al LLM."""
        from backend.llm.analyzer import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer(client=mock_client)
        result   = analyzer.compare([])
        assert result.trajectories_analyzed == 0
        mock_client.complete.assert_not_called()

    def test_rank_by_empty_list_no_llm_call(self, mock_client):
        """[FIX AN2] rank_by([]) retorna resultado vacío sin llamar al LLM."""
        from backend.llm.analyzer import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer(client=mock_client)
        result   = analyzer.rank_by([], "maximize salary")
        assert result.trajectories_analyzed == 0
        mock_client.complete.assert_not_called()

    def test_compare_calls_llm(self, mock_client, sample_trajectories):
        """compare() con trayectorias SÍ llama al LLM."""
        from backend.llm.analyzer import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer(client=mock_client)
        result   = analyzer.compare(sample_trajectories)
        mock_client.complete.assert_called_once()
        assert result.trajectories_analyzed == min(5, len(sample_trajectories))

    def test_rank_by_calls_llm(self, mock_client, sample_trajectories):
        """rank_by() con trayectorias SÍ llama al LLM."""
        from backend.llm.analyzer import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer(client=mock_client)
        result   = analyzer.rank_by(sample_trajectories, "maximize salary")
        mock_client.complete.assert_called_once()
        assert len(result.content) > 0

    def test_to_api_dict_no_prompt(self, mock_client, sample_trajectories):
        """[FIX AN3] to_api_dict() no expone el prompt al cliente."""
        from backend.llm.analyzer import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer(client=mock_client)
        result   = analyzer.compare(sample_trajectories)
        api_dict = result.to_api_dict()

        assert "prompt_used"  not in api_dict
        assert "_prompt_used" not in api_dict
        assert "content"       in api_dict
        assert "provider_used" in api_dict
        assert "analysis_type" in api_dict

    def test_prompt_stored_internally(self, mock_client, sample_trajectories):
        """[FIX AN3] El prompt sí se guarda internamente en _prompt_used."""
        from backend.llm.analyzer import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer(client=mock_client)
        result   = analyzer.compare(sample_trajectories)
        assert len(result._prompt_used) > 0

    def test_async_methods_exist(self, mock_client):
        """[FIX AN1] Los 4 métodos async deben existir en el analyzer."""
        from backend.llm.analyzer import TrajectoryAnalyzer
        import asyncio
        analyzer = TrajectoryAnalyzer(client=mock_client)
        for method_name in ("compare_async", "rank_by_async",
                            "analyze_async", "compare_terminals_async"):
            method = getattr(analyzer, method_name, None)
            assert method is not None, f"Método {method_name} no encontrado"
            assert asyncio.iscoroutinefunction(method), \
                f"{method_name} debe ser async"

    def test_provider_in_result(self, mock_client, sample_trajectories):
        """El provider_used se propaga al resultado."""
        from backend.llm.analyzer import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer(client=mock_client)
        result   = analyzer.compare(sample_trajectories)
        assert result.provider_used == "gemini"

    def test_compare_terminals(self, mock_client, sample_trajectories):
        """[FIX PR4] compare_terminals() usa build_terminal_comparison_prompt."""
        from backend.llm.analyzer import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer(client=mock_client)
        groups   = {
            "cto":       [sample_trajectories[0]],
            "ml_engineer": [sample_trajectories[1]],
        }
        result = analyzer.compare_terminals(groups, "maximize long-term income")
        mock_client.complete.assert_called_once()
        assert result.analysis_type == "terminal_comparison"
        assert result.trajectories_analyzed == 2

    def test_compare_terminals_empty(self, mock_client):
        """compare_terminals({}) no llama al LLM."""
        from backend.llm.analyzer import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer(client=mock_client)
        result   = analyzer.compare_terminals({}, "criterion")
        assert result.trajectories_analyzed == 0
        mock_client.complete.assert_not_called()


# ──────────────────────────────────────────────────────────────
# Tests de Prompts
# ──────────────────────────────────────────────────────────────

class TestPrompts:

    def test_no_ml_success_prob_in_format(self, sample_trajectories):
        """[FIX PR1] ml_success_prob eliminado — siempre era 50% falso."""
        from backend.llm.prompts import _format_trajectories
        formatted = _format_trajectories(sample_trajectories)
        assert "ml_success_prob" not in formatted
        assert "ML éxito" not in formatted

    def test_transition_probability_in_format(self, sample_trajectories):
        """[FIX PR1] transition_probability_score (dato real) presente."""
        from backend.llm.prompts import _format_trajectories
        formatted = _format_trajectories(sample_trajectories)
        # El campo transition_probability_score está en los scores de fixture
        assert "trans" in formatted.lower() or "45%" in formatted

    def test_labels_in_english(self, sample_trajectories):
        """[FIX PR5] Etiquetas en inglés para consistencia con el idioma del prompt."""
        from backend.llm.prompts import _format_trajectories
        formatted = _format_trajectories(sample_trajectories)
        assert "Final salary" in formatted
        assert "Growth"       in formatted
        assert "Demand"       in formatted
        assert "Salario"   not in formatted
        assert "Crecimiento" not in formatted

    def test_sim_section_absent_without_data(self, sample_trajectories):
        """[FIX PR2] sim_section no aparece cuando no hay datos de simulación."""
        from backend.llm.prompts import build_single_analysis_prompt
        prompt = build_single_analysis_prompt(sample_trajectories[0])
        assert "Monte Carlo" not in prompt

    def test_sim_section_present_with_data(self):
        """[FIX PR2] sim_section aparece cuando hay datos reales de simulación."""
        from backend.llm.prompts import build_single_analysis_prompt
        traj = _make_traj(("a", "b"), pareto_rank=0)
        traj.scores["sim_salary_p50"]    = 55_000
        traj.scores["sim_salary_p90"]    = 70_000
        traj.scores["sim_success_mean"]  = 0.75
        traj.scores["sim_risk_layoff"]   = 0.1
        traj.scores["sim_risk_burnout"]  = 0.12
        prompt = build_single_analysis_prompt(traj)
        assert "Monte Carlo" in prompt

    def test_ranking_prompt_with_precomputed_groups(self, sample_trajectories):
        """[FIX PR3] build_ranking_prompt acepta terminal_groups precalculado."""
        from backend.llm.prompts import build_ranking_prompt
        groups = {"cto": [sample_trajectories[0]], "ml_engineer": [sample_trajectories[1]]}
        prompt = build_ranking_prompt(sample_trajectories, "maximize salary",
                                      terminal_groups=groups)
        assert "maximize salary" in prompt
        assert "cto" in prompt.lower() or "CTO" in prompt

    def test_ranking_prompt_without_groups_still_works(self, sample_trajectories):
        """[FIX PR3] Retrocompatibilidad: funciona sin terminal_groups."""
        from backend.llm.prompts import build_ranking_prompt
        prompt = build_ranking_prompt(sample_trajectories, "maximize salary")
        assert "maximize salary" in prompt

    def test_terminal_comparison_prompt_usable(self, sample_trajectories):
        """[FIX PR4] build_terminal_comparison_prompt genera prompt correcto."""
        from backend.llm.prompts import build_terminal_comparison_prompt
        groups = {
            "cto":       [sample_trajectories[0]],
            "ml_engineer": [sample_trajectories[1]],
        }
        prompt = build_terminal_comparison_prompt(groups, "tech professional", "maximize impact")
        assert "SPANISH" in prompt
        assert "maximize impact" in prompt
        assert len(prompt) > 200

    def test_comparison_prompt_structure(self, sample_trajectories):
        """build_comparison_prompt tiene las 6 secciones requeridas."""
        from backend.llm.prompts import build_comparison_prompt
        prompt = build_comparison_prompt(sample_trajectories)
        assert "SPANISH" in prompt
        # Las 6 secciones están numeradas
        for i in range(1, 7):
            assert str(i) in prompt

    def test_prompts_respond_in_spanish(self, sample_trajectories):
        """Todos los prompts solicitan respuesta en español."""
        from backend.llm.prompts import (
            build_comparison_prompt, build_ranking_prompt, build_single_analysis_prompt,
        )
        for builder_result in [
            build_comparison_prompt(sample_trajectories),
            build_ranking_prompt(sample_trajectories, "test"),
            build_single_analysis_prompt(sample_trajectories[0]),
        ]:
            assert "SPANISH" in builder_result or "español" in builder_result.lower()
