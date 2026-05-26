"""tests/test_graph.py — Tests del grafo y scoring."""

import pytest
from core.graph import CareerGraph, Trajectory
from data.loader import load_career_graph


@pytest.fixture
def graph() -> CareerGraph:
    return CareerGraph(load_career_graph())


def test_graph_loads(graph):
    assert graph._g.number_of_nodes() == 12
    assert graph._g.number_of_edges() == 26


def test_successors_exist(graph):
    successors = graph.successors("junior_dev")
    assert len(successors) > 0
    assert "mid_dev" in successors


def test_score_trajectory_valid(graph):
    path = ("junior_dev", "mid_dev", "senior_dev")
    scores = graph.score_trajectory(path)
    assert "salary_growth" in scores
    assert scores["salary_growth"] > 0
    assert 0 <= scores["avg_risk"] <= 1


def test_score_trajectory_too_short(graph):
    """Trayectorias de 1 nodo no tienen scores."""
    scores = graph.score_trajectory(("junior_dev",))
    assert scores == {}


def test_iter_paths_no_cycles(graph):
    """Los caminos generados no deben contener ciclos."""
    paths = list(graph.iter_paths_from("junior_dev", max_depth=4))
    for path in paths:
        assert len(path) == len(set(path)), f"Ciclo detectado en: {path}"
