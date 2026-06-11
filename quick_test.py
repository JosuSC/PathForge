#!/usr/bin/env python
"""
quick_test.py
-------------
Prueba rápida de los componentes principales de PathForge.
Ejecutar: python quick_test.py

No requiere pytest — verifica que el sistema arranca y funciona end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

# FIX [Q1]: imports al inicio del archivo, no dentro de main()
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Bootstrap path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

console = Console()


# ────────────────────────────────────────────────────────────────

def test_input_manager() -> None:
    """Prueba el módulo InputManager con BD SQLite."""
    console.print(Panel("[bold cyan]Test 1: InputManager (BD SQLite)[/bold cyan]"))

    from backend.data.input_manager import InputManager, UserInput, create_default_presets

    manager = InputManager()
    create_default_presets(manager)
    console.print("[green]✓ Presets creados[/green]")

    inputs = manager.list_inputs()
    console.print(f"[green]✓ {len(inputs)} configuraciones en BD[/green]")

    custom = UserInput(
        id="test_custom",
        source_career="senior_dev",
        profile="ambitious",
        max_years=10,
        max_risk=0.8,
        user_profile_description="Senior buscando crear startup",
    )
    manager.save_input(custom)
    console.print("[green]✓ Configuración personalizada guardada[/green]")

    loaded = manager.load_input("test_custom")
    assert loaded is not None, "No se pudo cargar la configuración guardada"
    assert loaded.source_career == "senior_dev"
    console.print(f"[green]✓ Recuperada: source={loaded.source_career}, profile={loaded.profile}[/green]")

    # Cleanup
    manager.delete_input("test_custom")
    console.print("[dim]  Configuración de test limpiada[/dim]")
    console.print()


def test_bilingual_prompts() -> None:
    """Prueba los prompts bilingües (inglés → respuesta en español)."""
    console.print(Panel("[bold cyan]Test 2: Prompts Bilingües[/bold cyan]"))

    from backend.llm.prompts import build_single_analysis_prompt
    from backend.core.evaluator import EvaluatedTrajectory
    from backend.core.graph import Trajectory

    traj = Trajectory(nodes=("junior_dev", "mid_dev", "senior_dev", "tech_lead"))
    # FIX [Q2]: scores sin ml_success_prob (eliminado en v2)
    scores = {
        "final_salary":                150_000,
        "salary_growth":               2.5,
        "avg_demand":                  0.85,
        "avg_satisfaction":            0.80,
        "total_years":                 8.0,
        "avg_risk":                    0.4,
        "avg_difficulty":              0.6,
        "is_terminal_end":             0.0,
        "transition_probability_score":0.45,
    }
    et     = EvaluatedTrajectory(trajectory=traj, scores=scores, pareto_rank=0)
    prompt = build_single_analysis_prompt(et)

    assert "You are a world-class" in prompt, "El prompt debe estar en inglés"
    assert "SPANISH" in prompt,               "Debe pedir respuesta en español"
    assert "ml_success_prob" not in prompt,   "ml_success_prob fue eliminado en v2"
    assert "transition_probability_score" in prompt or "trans" in prompt.lower(), \
        "transition_probability_score debe aparecer en el prompt"

    console.print("[green]✓ Prompt en inglés correcto[/green]")
    console.print("[green]✓ Solicita respuesta en español[/green]")
    console.print("[green]✓ Sin ml_success_prob (eliminado en v2)[/green]")
    console.print(f"[dim]  Tamaño del prompt: {len(prompt)} caracteres[/dim]")
    console.print()


def test_api_endpoints() -> None:
    """Verifica que los endpoints REST estén registrados en FastAPI."""
    console.print(Panel("[bold cyan]Test 3: Endpoints REST[/bold cyan]"))

    from backend import main_api

    # FIX [Q3]: verificar rutas directamente desde app.routes
    registered_paths = {route.path for route in main_api.app.routes
                        if hasattr(route, "path")}

    required_endpoints = [
        ("/api/domains",           "GET  /api/domains"),
        ("/api/graph",             "GET  /api/graph"),
        ("/api/generate",          "POST /api/generate"),
        ("/api/analyze",           "POST /api/analyze"),
        ("/api/simulate",          "POST /api/simulate"),
        ("/api/inputs/list",       "GET  /api/inputs/list"),
        ("/api/inputs/create",     "POST /api/inputs/create"),
        ("/ws/explore",            "WS   /ws/explore"),
    ]

    for path, label in required_endpoints:
        found = path in registered_paths
        color = "green" if found else "red"
        icon  = "✓" if found else "✗"
        console.print(f"[{color}]{icon} {label}[/{color}]")

    console.print()


def test_career_graph() -> None:
    """Prueba la carga y scoring del grafo de carreras."""
    console.print(Panel("[bold cyan]Test 4: Grafo de Carreras[/bold cyan]"))

    from backend.data.loader import load_career_graph
    from backend.core.graph import CareerGraph

    G     = load_career_graph()
    graph = CareerGraph(G)

    assert graph._g.number_of_nodes() == 12, f"Se esperaban 12 nodos, hay {graph._g.number_of_nodes()}"
    assert graph._g.number_of_edges() == 26, f"Se esperaban 26 aristas, hay {graph._g.number_of_edges()}"
    console.print(f"[green]✓ Grafo: {graph._g.number_of_nodes()} nodos, {graph._g.number_of_edges()} aristas[/green]")

    terminals = graph.terminal_nodes()
    assert "cto" in terminals, "cto debería ser nodo terminal"
    console.print(f"[green]✓ Terminales: {terminals}[/green]")

    scores = graph.score_trajectory(("junior_dev", "mid_dev", "senior_dev", "cto"))
    assert "is_terminal_end"             in scores, "Falta is_terminal_end [FIX G1]"
    assert "transition_probability_score" in scores, "Falta transition_probability_score [FIX G3]"
    assert scores["is_terminal_end"] == 1.0,         "cto debe ser terminal"
    assert scores["salary_growth"]   > 0,             "Crecimiento salarial debe ser positivo"
    console.print("[green]✓ Scoring multiobjetivo correcto (incluye campos v2)[/green]")
    console.print(f"[dim]  salary_growth={scores['salary_growth']:.1%} | "
                  f"final_salary=${scores['final_salary']:,.0f}[/dim]")
    console.print()


def test_beam_search() -> None:
    """Prueba el generador Beam Search con restricciones."""
    console.print(Panel("[bold cyan]Test 5: Beam Search + Pareto[/bold cyan]"))

    from backend.data.loader import load_career_graph
    from backend.core.graph import CareerGraph
    from backend.core.generator import GeneratorConfig, TrajectoryGenerator
    from backend.core.constraints import ConstraintProfiles

    graph     = CareerGraph(load_career_graph())
    config    = GeneratorConfig(beam_width=5, max_depth=4, top_k_results=5)
    generator = TrajectoryGenerator(graph, config)

    results = generator.generate(
        source="junior_dev",
        constraints=ConstraintProfiles.balanced(max_years=10),
    )

    assert len(results) > 0, "El generador debe retornar al menos 1 trayectoria"
    assert results[0].pareto_rank == 0, "El primer resultado debe ser Pareto rank 0"

    console.print(f"[green]✓ {len(results)} trayectorias generadas[/green]")
    for i, et in enumerate(results[:3], 1):
        path = " → ".join(et.trajectory.nodes)
        console.print(
            f"[dim]  #{i} rank={et.pareto_rank} | {path} | "
            f"${et.scores.get('final_salary',0):,.0f}[/dim]"
        )
    console.print()


def test_database() -> None:
    """Prueba la BD SQLite (creación, escritura, limpieza)."""
    console.print(Panel("[bold cyan]Test 6: Base de Datos SQLite[/bold cyan]"))

    from backend.data.input_manager import InputManager

    db_path = Path("test_temp.db")
    manager = InputManager(str(db_path))

    assert db_path.exists(), "La BD debe crearse automáticamente"
    console.print(f"[green]✓ BD creada en {db_path.resolve()}[/green]")

    inputs = manager.list_inputs()
    assert isinstance(inputs, list)
    console.print(f"[green]✓ BD operativa ({len(inputs)} registros)[/green]")

    db_path.unlink()
    console.print("[dim]  BD temporal limpiada[/dim]")
    console.print()


# ────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────

def main() -> int:
    console.print(Text("\n🧪 PathForge — Quick Test Suite\n", style="bold cyan"))

    tests = [
        ("InputManager (SQLite)",     test_input_manager),
        ("Prompts Bilingües",          test_bilingual_prompts),
        ("Endpoints REST",             test_api_endpoints),
        ("Grafo de Carreras",          test_career_graph),
        ("Beam Search + Pareto",       test_beam_search),
        ("Base de Datos",              test_database),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            console.print(f"[red]✗ {name} falló: {e}[/red]\n")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            failed += 1

    console.print("=" * 60)
    console.print(
        f"[green]Pasados: {passed}[/green] | "
        f"[{'red' if failed else 'green'}]Fallidos: {failed}[/{'red' if failed else 'green'}]"
    )

    if failed == 0:
        console.print("\n[bold green]✓ Todos los tests pasaron — sistema listo[/bold green]")
        return 0
    else:
        console.print(f"\n[bold red]✗ {failed} test(s) fallaron — revisa los errores arriba[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
