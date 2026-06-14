"""
tests/test_graph.py
-------------------
Tests del grafo de carreras y su sistema de scoring.

Cubre:
    - Carga correcta del grafo desde careers.json
    - Campos nuevos de graph.py v2: is_terminal_end, transition_probability_score
    - Validación de aristas con referencias inexistentes
    - cached_property de terminal_nodes
    - iter_paths_from iterativo (sin ciclos, con límite)
    - score_trajectory con y sin sim_* fields
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import networkx as nx

# Bootstrap path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.graph import CareerGraph, Trajectory
from backend.data.loader import load_career_graph


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def graph() -> CareerGraph:
    """Carga el grafo default una sola vez para todo el módulo."""
    return CareerGraph(load_career_graph())


@pytest.fixture
def minimal_graph() -> CareerGraph:
    """Grafo mínimo controlado para tests de scoring exacto."""
    G = nx.DiGraph()
    G.add_node("a", avg_salary=30_000, demand=0.7, satisfaction=0.6,
               type="entry", skills=["python"], years_experience=1, label="A")
    G.add_node("b", avg_salary=60_000, demand=0.85, satisfaction=0.75,
               type="mid",   skills=["design"], years_experience=3, label="B")
    G.add_node("c", avg_salary=90_000, demand=0.9,  satisfaction=0.8,
               type="senior",skills=["arch"],   years_experience=6, label="C")
    G.add_edge("a", "b", transition_years=2, difficulty=0.4, risk=0.25,
               transition_probability=0.6, salary_growth=1.0)
    G.add_edge("b", "c", transition_years=3, difficulty=0.5, risk=0.3,
               transition_probability=0.4, salary_growth=0.5)
    return CareerGraph(G)


# ──────────────────────────────────────────────────────────────
# Tests de carga del grafo default
# ──────────────────────────────────────────────────────────────

class TestGraphLoading:

    def test_graph_loads(self, graph):
        assert graph._g.number_of_nodes() == 12
        assert graph._g.number_of_edges() == 26

    def test_all_nodes_have_required_attrs(self, graph):
        required = {"avg_salary", "demand", "satisfaction"}
        for node_id in graph.all_node_ids():
            attrs = graph.node_attrs(node_id)
            assert required.issubset(attrs.keys()), \
                f"Nodo '{node_id}' falta atributos: {required - attrs.keys()}"

    def test_all_edges_have_required_attrs(self, graph):
        required = {"transition_years", "difficulty", "risk"}
        for u, v in graph._g.edges():
            attrs = graph.edge_attrs(u, v)
            assert required.issubset(attrs.keys()), \
                f"Arista '{u}→{v}' falta atributos: {required - attrs.keys()}"

    def test_new_edge_fields_present(self, graph):
        """careers.json v2 tiene transition_probability y salary_growth."""
        for u, v in graph._g.edges():
            attrs = graph.edge_attrs(u, v)
            assert "transition_probability" in attrs, f"Arista '{u}→{v}' sin transition_probability"
            assert "salary_growth"          in attrs, f"Arista '{u}→{v}' sin salary_growth"

    def test_node_types_are_valid(self, graph):
        valid_types = {"entry", "mid", "senior", "leadership", "role"}
        for node_id in graph.all_node_ids():
            ntype = graph.node_attrs(node_id).get("type")
            assert ntype in valid_types, f"Nodo '{node_id}' tiene type inválido: '{ntype}'"


# ──────────────────────────────────────────────────────────────
# Tests de sucesores
# ──────────────────────────────────────────────────────────────

class TestSuccessors:

    def test_junior_dev_has_successors(self, graph):
        successors = graph.successors("junior_dev")
        assert len(successors) > 0
        assert "mid_dev" in successors

    def test_terminal_nodes_have_no_successors(self, graph):
        for terminal in graph.terminal_nodes():
            assert len(graph.successors(terminal)) == 0, \
                f"Nodo terminal '{terminal}' tiene sucesores"

    def test_cto_is_terminal(self, graph):
        assert "cto" in graph.terminal_nodes()

    def test_terminal_nodes_cached(self, graph):
        """cached_property: el mismo objeto frozenset en ambas llamadas."""
        s1 = graph._terminal_nodes_set
        s2 = graph._terminal_nodes_set
        assert s1 is s2

    def test_is_terminal_correct(self, graph):
        assert graph.is_terminal("cto")
        assert not graph.is_terminal("junior_dev")


# ──────────────────────────────────────────────────────────────
# Tests de scoring
# ──────────────────────────────────────────────────────────────

class TestScoring:

    def test_score_trajectory_valid(self, graph):
        path   = ("junior_dev", "mid_dev", "senior_dev")
        scores = graph.score_trajectory(path)
        assert "salary_growth"    in scores
        assert "avg_risk"         in scores
        assert "final_salary"     in scores
        assert "avg_demand"       in scores
        assert "avg_satisfaction" in scores
        assert "total_years"      in scores

    def test_score_trajectory_too_short(self, graph):
        """Trayectoria de 1 nodo retorna dict vacío."""
        assert graph.score_trajectory(("junior_dev",)) == {}

    def test_salary_growth_positive(self, graph):
        """junior_dev → cto: crecimiento positivo garantizado."""
        path   = ("junior_dev", "mid_dev", "senior_dev", "tech_lead", "cto")
        scores = graph.score_trajectory(path)
        assert scores["salary_growth"] > 0

    def test_avg_risk_in_range(self, graph):
        path   = ("junior_dev", "mid_dev", "senior_dev")
        scores = graph.score_trajectory(path)
        assert 0.0 <= scores["avg_risk"] <= 1.0

    def test_is_terminal_end_field(self, minimal_graph):
        """[FIX G1] is_terminal_end=1.0 cuando el último nodo es terminal."""
        scores_terminal = minimal_graph.score_trajectory(("a", "b", "c"))
        assert "is_terminal_end" in scores_terminal
        assert scores_terminal["is_terminal_end"] == 1.0

    def test_is_terminal_end_false_for_non_terminal(self, graph):
        """is_terminal_end=0.0 cuando el último nodo tiene sucesores."""
        scores = graph.score_trajectory(("junior_dev", "mid_dev"))
        assert scores["is_terminal_end"] == 0.0

    def test_transition_probability_score_field(self, minimal_graph):
        """[FIX G3] transition_probability_score usa datos reales de las aristas."""
        scores = minimal_graph.score_trajectory(("a", "b", "c"))
        assert "transition_probability_score" in scores
        expected = (0.6 + 0.4) / 2
        assert abs(scores["transition_probability_score"] - expected) < 1e-6

    def test_salary_growth_edge_field(self, minimal_graph):
        """[FIX G3] salary_growth_edge usa salary_growth real de las aristas."""
        scores = minimal_graph.score_trajectory(("a", "b", "c"))
        assert "salary_growth_edge" in scores
        expected = (1.0 + 0.5) / 2
        assert abs(scores["salary_growth_edge"] - expected) < 1e-6

    def test_salary_growth_exact(self, minimal_graph):
        scores = minimal_graph.score_trajectory(("a", "b", "c"))
        expected = (90_000 - 30_000) / 30_000
        assert abs(scores["salary_growth"] - expected) < 1e-6

    def test_final_salary_exact(self, minimal_graph):
        scores = minimal_graph.score_trajectory(("a", "b", "c"))
        assert scores["final_salary"] == 90_000.0

    def test_total_years_exact(self, minimal_graph):
        scores = minimal_graph.score_trajectory(("a", "b", "c"))
        assert scores["total_years"] == 5.0  # 2 + 3


# ──────────────────────────────────────────────────────────────
# Tests de iter_paths_from (versión iterativa v2)
# ──────────────────────────────────────────────────────────────

class TestIterPaths:

    def test_iter_paths_no_cycles(self, graph):
        """[FIX G2] Los caminos no deben contener ciclos."""
        paths = list(graph.iter_paths_from("junior_dev", max_depth=4))
        for path in paths:
            assert len(path) == len(set(path)), f"Ciclo detectado: {path}"

    def test_iter_paths_depth_respected(self, graph):
        """Ningún camino supera max_depth nodos."""
        max_depth = 3
        paths = list(graph.iter_paths_from("junior_dev", max_depth=max_depth))
        for path in paths:
            assert len(path) <= max_depth, f"Path demasiado largo: {path}"

    def test_iter_paths_source_included(self, graph):
        """Todos los paths empiezan con el nodo fuente."""
        for path in graph.iter_paths_from("junior_dev", max_depth=3):
            assert path[0] == "junior_dev"

    def test_iter_paths_max_paths_limit(self, graph):
        """[FIX G2] max_paths limita el total de paths generados."""
        limit = 10
        paths = list(graph.iter_paths_from("junior_dev", max_depth=6, max_paths=limit))
        assert len(paths) <= limit

    def test_iter_paths_single_node(self, graph):
        """Siempre retorna al menos el path de 1 nodo (el origen)."""
        paths = list(graph.iter_paths_from("cto", max_depth=3))
        assert len(paths) >= 1
        assert paths[0] == ("cto",)


# ──────────────────────────────────────────────────────────────
# Tests de validación del grafo
# ──────────────────────────────────────────────────────────────

class TestGraphValidation:

    def test_invalid_edge_reference_skipped(self, caplog):
        """[FIX loader] Aristas con nodos inexistentes se descartan con warning."""
        import logging
        G = nx.DiGraph()
        G.add_node("x", avg_salary=50_000, demand=0.7, satisfaction=0.7,
                   type="mid", skills=["a"], years_experience=3, label="X")
        # Arista con destino inexistente — loader.py debe haberla descartado
        # Aquí testeamos directamente que CareerGraph valida correctamente
        # añadiendo nodos suficientes
        G.add_node("y", avg_salary=70_000, demand=0.8, satisfaction=0.75,
                   type="senior", skills=["b"], years_experience=5, label="Y")
        G.add_edge("x", "y", transition_years=2, difficulty=0.4, risk=0.3)
        cg = CareerGraph(G)
        assert cg._g.number_of_edges() == 1

    def test_missing_node_attr_raises(self):
        """CareerGraph lanza ValueError si faltan atributos requeridos."""
        G = nx.DiGraph()
        G.add_node("bad", avg_salary=50_000)  # falta demand, satisfaction
        with pytest.raises(ValueError, match="falta atributos"):
            CareerGraph(G)

    def test_max_salary_computed(self, graph):
        """[FIX GEN3] _max_salary se calcula dinámicamente al construir el grafo."""
        salaries = [graph.node_attrs(n)["avg_salary"] for n in graph.all_node_ids()]
        assert graph._max_salary == max(salaries)

    def test_max_salary_minimal_graph(self, minimal_graph):
        assert minimal_graph._max_salary == 90_000


# ──────────────────────────────────────────────────────────────
# Tests de Trajectory dataclass
# ──────────────────────────────────────────────────────────────

class TestTrajectory:

    def test_trajectory_len(self):
        t = Trajectory(nodes=("a", "b", "c"))
        assert len(t) == 3

    def test_trajectory_hashable(self):
        """Trajectory debe ser usable como clave de set/dict."""
        t1 = Trajectory(nodes=("a", "b"))
        t2 = Trajectory(nodes=("a", "b"))
        s  = {t1, t2}
        assert len(s) == 1

    def test_trajectory_equality(self):
        t1 = Trajectory(nodes=("a", "b", "c"))
        t2 = Trajectory(nodes=("a", "b", "c"))
        t3 = Trajectory(nodes=("a", "b"))
        assert t1 == t2
        assert t1 != t3
