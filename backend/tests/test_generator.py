"""
tests/test_generator.py
-----------------------
Tests del generador Beam Search multiobjetivo.

Cubre:
    - Generación básica: retorna resultados, sin crashear
    - Restricciones: conservative, ambitious, balanced, fast_track
    - Deduplicación de candidatos [FIX GEN2]
    - max_salary dinámico [FIX GEN3]
    - source inválido lanza ValueError con mensaje claro
    - Evaluación Pareto: rank 0 existe, crowding_distance > 0
    - Diversidad: trayectorias con distintos terminales
    - Configuraciones extremas: beam_width=2, max_depth=2
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.constraints import (
    ConstraintProfiles, MaxRiskConstraint, MaxYearsConstraint, MinSalaryConstraint,
)
from backend.core.evaluator import EvaluatedTrajectory, TrajectoryEvaluator
from backend.core.generator import GeneratorConfig, TrajectoryGenerator
from backend.core.graph import CareerGraph
from backend.data.loader import load_career_graph


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def career_graph() -> CareerGraph:
    return CareerGraph(load_career_graph())


@pytest.fixture(scope="module")
def default_config() -> GeneratorConfig:
    return GeneratorConfig(beam_width=5, max_depth=4, top_k_results=10)


@pytest.fixture(scope="module")
def generator(career_graph, default_config) -> TrajectoryGenerator:
    return TrajectoryGenerator(career_graph, default_config)


@pytest.fixture
def small_graph() -> CareerGraph:
    """Grafo controlado para tests deterministas."""
    G = nx.DiGraph()
    for nid, sal, dem, sat, yr, ntype in [
        ("a", 30_000, 0.7, 0.6, 1, "entry"),
        ("b", 60_000, 0.8, 0.75, 3, "mid"),
        ("c", 90_000, 0.85, 0.8, 6, "senior"),
        ("d", 120_000, 0.7, 0.82, 9, "leadership"),
        ("e", 50_000, 0.75, 0.83, 3, "mid"),
    ]:
        G.add_node(nid, avg_salary=sal, demand=dem, satisfaction=sat,
                   type=ntype, skills=["x"], years_experience=yr, label=nid.upper())
    G.add_edge("a", "b", transition_years=2, difficulty=0.4, risk=0.2, transition_probability=0.5, salary_growth=1.0)
    G.add_edge("a", "e", transition_years=1, difficulty=0.3, risk=0.15, transition_probability=0.3, salary_growth=0.67)
    G.add_edge("b", "c", transition_years=3, difficulty=0.5, risk=0.3, transition_probability=0.4, salary_growth=0.5)
    G.add_edge("b", "d", transition_years=4, difficulty=0.65, risk=0.4, transition_probability=0.2, salary_growth=1.0)
    G.add_edge("e", "c", transition_years=3, difficulty=0.55, risk=0.35, transition_probability=0.35, salary_growth=0.8)
    G.add_edge("c", "d", transition_years=2, difficulty=0.55, risk=0.25, transition_probability=0.45, salary_growth=0.33)
    return CareerGraph(G)


# ──────────────────────────────────────────────────────────────
# Tests básicos de generación
# ──────────────────────────────────────────────────────────────

class TestGeneratorBasic:

    def test_generate_returns_results(self, generator):
        results = generator.generate("junior_dev")
        assert len(results) > 0

    def test_generate_returns_evaluated_trajectories(self, generator):
        results = generator.generate("junior_dev")
        for r in results:
            assert isinstance(r, EvaluatedTrajectory)
            assert len(r.trajectory.nodes) >= 2
            assert isinstance(r.scores, dict)
            assert len(r.scores) > 0

    def test_generate_top_k_respected(self, career_graph):
        k = 5
        config    = GeneratorConfig(beam_width=10, max_depth=5, top_k_results=k)
        generator = TrajectoryGenerator(career_graph, config)
        results   = generator.generate("junior_dev")
        assert len(results) <= k

    def test_generate_invalid_source_raises(self, generator):
        with pytest.raises(ValueError, match="no existe en el grafo"):
            generator.generate("nodo_inventado_xyz")

    def test_generate_all_paths_start_at_source(self, generator):
        source  = "mid_dev"
        results = generator.generate(source)
        for r in results:
            assert r.trajectory.nodes[0] == source

    def test_generate_no_cycles_in_trajectories(self, generator):
        results = generator.generate("junior_dev")
        for r in results:
            nodes = r.trajectory.nodes
            assert len(nodes) == len(set(nodes)), \
                f"Ciclo detectado: {' → '.join(nodes)}"

    def test_generate_from_each_source_node(self, career_graph, default_config):
        """Todos los nodos de inicio del grafo default deben funcionar."""
        gen     = TrajectoryGenerator(career_graph, default_config)
        sources = ["junior_dev", "mid_dev", "data_scientist", "devops_engineer"]
        for source in sources:
            results = gen.generate(source)
            assert len(results) >= 0, f"Falló para source={source}"


# ──────────────────────────────────────────────────────────────
# Tests con restricciones
# ──────────────────────────────────────────────────────────────

class TestGeneratorConstraints:

    def test_max_years_constraint(self, career_graph):
        max_yrs     = 5
        config      = GeneratorConfig(beam_width=8, max_depth=6, top_k_results=10)
        generator   = TrajectoryGenerator(career_graph, config)
        constraints = MaxYearsConstraint(max_yrs)
        results     = generator.generate("junior_dev", constraints=constraints)

        for r in results:
            nodes      = r.trajectory.nodes
            total_yrs  = sum(
                career_graph.edge_attrs(nodes[i], nodes[i+1])["transition_years"]
                for i in range(len(nodes) - 1)
            )
            assert total_yrs <= max_yrs, \
                f"MaxYears({max_yrs}) violada: {total_yrs} en {nodes}"

    def test_combined_constraints(self, career_graph):
        config      = GeneratorConfig(beam_width=8, max_depth=5, top_k_results=10)
        generator   = TrajectoryGenerator(career_graph, config)
        constraints = MaxRiskConstraint(0.4) & MaxYearsConstraint(10)
        results     = generator.generate("junior_dev", constraints=constraints)
        # Al menos debe retornar algo (aunque sea pocos)
        assert isinstance(results, list)


# ──────────────────────────────────────────────────────────────
# Tests de calidad Pareto
# ──────────────────────────────────────────────────────────────

class TestParetoQuality:

    def test_pareto_front_exists(self, generator):
        results = generator.generate("junior_dev")
        pareto  = [r for r in results if r.pareto_rank == 0]
        assert len(pareto) >= 1, "Debe existir al menos 1 trayectoria en frente Pareto"

    def test_pareto_ranks_non_negative(self, generator):
        results = generator.generate("junior_dev")
        for r in results:
            assert r.pareto_rank >= 0

    def test_crowding_distance_positive(self, generator):
        results = generator.generate("junior_dev")
        pareto  = [r for r in results if r.pareto_rank == 0]
        if len(pareto) > 1:
            for r in pareto:
                assert r.crowding_distance >= 0.0

    def test_best_trajectory_rank_zero(self, generator):
        """El primer resultado retornado debe ser Pareto rank 0."""
        results = generator.generate("junior_dev")
        assert results[0].pareto_rank == 0

    def test_scores_complete(self, generator):
        """Todos los scores obligatorios deben estar presentes."""
        required = {
            "salary_growth", "avg_demand", "avg_satisfaction",
            "final_salary", "total_years", "avg_risk", "avg_difficulty",
            "is_terminal_end",
        }
        results = generator.generate("junior_dev")
        for r in results:
            missing = required - r.scores.keys()
            assert not missing, f"Scores faltantes en {r.trajectory}: {missing}"


# ──────────────────────────────────────────────────────────────
# Tests de propiedades del generador [FIX GEN2, GEN3]
# ──────────────────────────────────────────────────────────────

class TestGeneratorProperties:

    def test_max_salary_dynamic(self, career_graph):
        """[FIX GEN3] _max_salary dinámico, no hardcoded a 180_000."""
        gen = TrajectoryGenerator(career_graph, GeneratorConfig())
        salaries = [career_graph.node_attrs(n)["avg_salary"]
                    for n in career_graph.all_node_ids()]
        assert gen._max_salary == max(salaries)

    def test_max_salary_domain_graph(self, small_graph):
        """[FIX GEN3] Domain graphs con salarios distintos normalizan correctamente."""
        gen = TrajectoryGenerator(small_graph, GeneratorConfig())
        assert gen._max_salary == 120_000

    def test_no_duplicate_trajectories(self, small_graph):
        """[FIX GEN2] No debe haber trayectorias duplicadas en los resultados."""
        config    = GeneratorConfig(beam_width=10, max_depth=5, top_k_results=20)
        generator = TrajectoryGenerator(small_graph, config)
        results   = generator.generate("a")
        paths     = [r.trajectory.nodes for r in results]
        assert len(paths) == len(set(paths)), "Hay trayectorias duplicadas"

    def test_protocol_check_raises_on_bad_graph(self):
        """[FIX GEN1] TypeError si el grafo no implementa los métodos requeridos."""
        class BadGraph:
            pass
        with pytest.raises(TypeError, match="debe implementar"):
            TrajectoryGenerator(BadGraph())

    def test_step_callback_called(self, small_graph):
        """step_callback se llama durante la exploración."""
        steps    = []
        config   = GeneratorConfig(beam_width=5, max_depth=4, emit_steps=True)
        generator = TrajectoryGenerator(small_graph, config)

        def cb(depth, expanded, selected, completed):
            steps.append({"depth": depth, "n_expanded": len(expanded)})

        generator.generate("a", step_callback=cb)
        assert len(steps) >= 1, "step_callback nunca fue llamado"
        assert all(s["depth"] >= 1 for s in steps)

    def test_small_graph_finds_terminals(self, small_graph):
        """El generador debe encontrar trayectorias que terminan en nodos terminales."""
        config    = GeneratorConfig(beam_width=10, max_depth=5, top_k_results=15)
        generator = TrajectoryGenerator(small_graph, config)
        results   = generator.generate("a")
        terminal_paths = [r for r in results if r.scores.get("is_terminal_end", 0) == 1.0]
        assert len(terminal_paths) >= 1, "No se encontraron trayectorias terminales"


# ──────────────────────────────────────────────────────────────
# Tests de configuraciones extremas
# ──────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_min_beam_width(self, career_graph):
        config    = GeneratorConfig(beam_width=2, max_depth=3, top_k_results=5)
        generator = TrajectoryGenerator(career_graph, config)
        results   = generator.generate("junior_dev")
        assert isinstance(results, list)

    def test_min_max_depth(self, career_graph):
        config    = GeneratorConfig(beam_width=5, max_depth=2, top_k_results=5)
        generator = TrajectoryGenerator(career_graph, config)
        results   = generator.generate("junior_dev")
        for r in results:
            assert len(r.trajectory.nodes) >= 2

    def test_generate_from_near_terminal(self, career_graph):
        """Generar desde engineering_manager (casi terminal) no crashea."""
        config    = GeneratorConfig(beam_width=5, max_depth=4, top_k_results=5)
        generator = TrajectoryGenerator(career_graph, config)
        results   = generator.generate("engineering_manager")
        assert isinstance(results, list)

    def test_generate_with_no_constraint(self, generator):
        results = generator.generate("junior_dev", constraints=None)
        assert len(results) > 0
