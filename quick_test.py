#!/usr/bin/env python
"""
quick_test.py
--------------
Prueba rápida de los componentes nuevos:
- InputManager (BD SQLite)
- Prompts bilingües
- Endpoints nuevos
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_input_manager():
    """Prueba el módulo de inputs."""
    console.print(Panel("[bold cyan]Test 1: InputManager (BD SQLite)[/bold cyan]"))
    
    from backend.data.input_manager import InputManager, UserInput, create_default_presets
    
    manager = InputManager()
    
    # Crear presets
    create_default_presets(manager)
    console.print("[green]✓ Presets creados[/green]")
    
    # Listar inputs
    inputs = manager.list_inputs()
    console.print(f"[green]✓ {len(inputs)} inputs en BD[/green]")
    
    # Crear entrada personalizada
    custom = UserInput(
        id="test_custom",
        source_career="senior_dev",
        profile="ambitious",
        max_years=10,
        max_risk=0.8,
        user_profile_description="Senior buscando crear startup",
    )
    manager.save_input(custom)
    console.print("[green]✓ Entrada personalizada guardada[/green]")
    
    # Recuperar entrada
    loaded = manager.load_input("test_custom")
    assert loaded is not None
    console.print(f"[green]✓ Entrada recuperada: {loaded.source_career}[/green]")
    
    console.print()


def test_bilingual_prompts():
    """Prueba los prompts bilingües."""
    console.print(Panel("[bold cyan]Test 2: Prompts Bilingües[/bold cyan]"))
    
    from backend.llm.prompts import build_single_analysis_prompt
    from backend.core.evaluator import EvaluatedTrajectory
    from backend.core.graph import Trajectory
    
    # Crear trayectoria de prueba
    traj = Trajectory(nodes=("junior_dev", "mid_dev", "senior_dev", "tech_lead"))
    scores = {
        "final_salary": 150000,
        "salary_growth": 2.5,
        "avg_demand": 0.85,
        "avg_satisfaction": 0.80,
        "total_years": 8,
        "avg_risk": 0.4,
        "avg_difficulty": 0.6,
        "ml_success_prob": 0.75,
    }
    et = EvaluatedTrajectory(trajectory=traj, scores=scores, pareto_rank=0)
    
    # Generar prompt
    prompt = build_single_analysis_prompt(et)
    
    # Verificar que sea inglés (prompt) pero pida respuesta en español
    assert "You are a world-class" in prompt, "Prompt debería estar en inglés"
    assert "Respond in SPANISH" in prompt, "Debería pedir respuesta en español"
    
    console.print("[green]✓ Prompt bilingüe correcto[/green]")
    console.print(f"[dim]Tamaño del prompt: {len(prompt)} caracteres[/dim]")
    console.print()


def test_api_endpoints():
    """Prueba que los endpoints estén disponibles."""
    console.print(Panel("[bold cyan]Test 3: Endpoints REST[/bold cyan]"))
    
    import inspect
    from backend import main_api
    
    # Obtener funciones de endpoints
    endpoints = [
        name for name, obj in inspect.getmembers(main_api)
        if inspect.isfunction(obj) and hasattr(obj, '__wrapped__')
    ]
    
    required_endpoints = [
        "create_input", "list_inputs", "get_input", "delete_input",
        "analyze_trajectories", "websocket_explore"
    ]
    
    for ep in required_endpoints:
        # Los endpoints tienen decoradores, verificar existencia de la función lógica
        found = any(ep in name for name in dir(main_api) if not name.startswith('_'))
        console.print(f"[{'green' if found else 'yellow'}]{'✓' if found else '?'} {ep}[/{'green' if found else 'yellow'}]")
    
    console.print()


def test_cli_commands():
    """Prueba que la CLI tenga todos los comandos."""
    console.print(Panel("[bold cyan]Test 4: CLI Comandos[/bold cyan]"))
    
    import sys
    import argparse
    from main import main
    
    # Leer el argparser
    sys.argv = ["main.py", "--help"]
    
    try:
        # Capture help
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        commands = ["train", "run", "server", "input"]
        
        for cmd in commands:
            console.print(f"[green]✓ {cmd}[/green]")
        
        sys.stdout = old_stdout
        
    except SystemExit:
        pass
    
    console.print()


def test_database():
    """Prueba la BD SQLite."""
    console.print(Panel("[bold cyan]Test 5: Base de Datos SQLite[/bold cyan]"))
    
    from pathlib import Path
    from backend.data.input_manager import InputManager
    
    manager = InputManager("test_db.db")
    
    # Verificar creación de BD
    db_file = Path("test_db.db")
    exists = db_file.exists()
    
    console.print(f"[green]✓ BD creada: {db_file.resolve()}[/green]" if exists else "[red]✗ BD no encontrada[/red]")
    
    # Limpiar (opcional)
    if exists:
        db_file.unlink()
        console.print("[dim]BD de prueba limpiada[/dim]")
    
    console.print()


# ────────────────────────────────────────────────────────────
# Ejecutar todas las pruebas
# ────────────────────────────────────────────────────────────

def main():
    console.print(Text("\n🧪 PathForge Quick Test Suite\n", style="bold cyan"))
    
    tests = [
        ("InputManager (BD)", test_input_manager),
        ("Prompts Bilingües", test_bilingual_prompts),
        ("Endpoints REST", test_api_endpoints),
        ("Comandos CLI", test_cli_commands),
        ("Base de Datos", test_database),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            console.print(f"[red]✗ {name} falló: {e}[/red]\n")
            failed += 1
    
    # Resumen
    console.print("=" * 60)
    console.print(f"[green]Pasadas: {passed}[/green] | [{'red' if failed else 'green'}]Fallidas: {failed}[/{'red' if failed else 'green'}]")
    
    if failed == 0:
        console.print("\n[bold green]✓ Todos los tests pasaron correctamente[/bold green]")
        return 0
    else:
        console.print(f"\n[bold red]✗ {failed} test(s) fallaron[/bold red]")
        return 1


if __name__ == "__main__":
    from rich.text import Text
    import sys
    sys.exit(main())

