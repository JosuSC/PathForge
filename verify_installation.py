#!/usr/bin/env python
"""
verify_installation.py
----------------------
Verifica que todos los componentes de PathForge estén instalados correctamente.
Ejecutar: python verify_installation.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

# ────────────────────────────────────────────────────────────
# Verificaciones
# ────────────────────────────────────────────────────────────

checks = []


def check(name: str, func) -> None:
    """Ejecuta una verificación."""
    try:
        result = func()
        checks.append((name, "✓", "green", result or "OK"))
    except Exception as e:
        checks.append((name, "✗", "red", str(e)[:50]))


# 1. Dependencias Python
def verify_imports():
    required_imports = [
        "fastapi", "uvicorn", "websockets", "pydantic",
        "sklearn", "networkx", "numpy", "pandas",
        "loguru", "rich", "dotenv"
    ]

    optional_imports = ["google.genai", "anthropic", "openai"]

    missing = []
    for imp in required_imports:
        try:
            __import__(imp)
        except ImportError:
            missing.append(imp)

    optional_missing = []
    for imp in optional_imports:
        try:
            __import__(imp)
        except ImportError:
            optional_missing.append(imp)

    if missing:
        raise ImportError(f"Missing required: {', '.join(missing[:4])}")

    if optional_missing:
        return (
            f"{len(required_imports)}/{len(required_imports)} required OK | "
            f"optional missing: {', '.join(optional_missing)}"
        )

    return f"{len(required_imports)}/{len(required_imports)} required imports OK"


# 2. Módulos del proyecto
def verify_core_modules():
    modules = [
        "backend.core.simulation",
        "backend.core.generator",
        "backend.core.graph",
        "backend.data.loader",
        "backend.data.input_manager",
        "backend.llm.prompts",
        "backend.llm.analyzer",
        "backend.llm.client",
    ]
    ok = 0
    for mod in modules:
        try:
            __import__(mod)
            ok += 1
        except:
            pass
    return f"{ok}/{len(modules)} modules found"


# 3. Archivos clave
def verify_files():
    files = [
        "backend/main_api.py",
        "backend/data/input_manager.py",
        "main.py",
        "frontend/index.html",
        "frontend/js/websocket.js",
    ]
    root = Path(__file__).parent
    found = sum(1 for f in files if (root / f).exists())
    return f"{found}/{len(files)} files exist"


# 4. Base de datos
def verify_database():
    from backend.data.input_manager import InputManager
    manager = InputManager()
    inputs = manager.list_inputs()
    return f"{len(inputs)} inputs in DB"


# 5. Configuración LLM
def verify_llm_setup():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    env_keys = [k for k in os.environ if k.startswith("LLM_")]
    return f"{len(env_keys)} LLM keys configured" if env_keys else "⚠️  No LLM keys (check .env)"


# 6. Grafo de carrera
def verify_graph():
    from backend.data.loader import load_career_graph
    from backend.core.graph import CareerGraph
    G = load_career_graph()
    graph = CareerGraph(G)
    nodes = graph.all_node_ids()

    # Terminal = nodo sin salidas (out-degree 0)
    terminal_count = sum(1 for n in nodes if len(graph.successors(n)) == 0)
    return f"{len(nodes)} career nodes, {terminal_count} terminals"


# 7. Predictor ML
def verify_predictor():
    from backend.core.scorer import CareerOutcomePredictor
    p = CareerOutcomePredictor.load_or_train()
    return f"Model score: {round(p.cv_score, 3)}"


# ────────────────────────────────────────────────────────────
# Ejecutar verificaciones
# ────────────────────────────────────────────────────────────

def main():
    console.print(
        Text("\n🔍 PathForge Installation Verification\n", style="bold cyan")
    )

    check("Python Imports", verify_imports)
    check("Core Modules", verify_core_modules)
    check("Project Files", verify_files)
    check("Database (SQLite)", verify_database)
    check("LLM Configuration", verify_llm_setup)
    check("Career Graph", verify_graph)
    check("ML Predictor", verify_predictor)

    # Mostrar tabla
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Details", style="dim")

    failed = 0
    for name, status, color, details in checks:
        if status == "✗":
            failed += 1
        table.add_row(name, Text(status, style=color), details)

    console.print(table)

    if failed == 0:
        console.print("\n[green]✓ All components verified successfully![/green]")
        console.print("[dim]Ready to use: python main.py train[/dim]\n")
        return 0
    else:
        console.print(f"\n[red]✗ {failed} component(s) failed verification[/red]")
        console.print("[yellow]Check errors above and fix before proceeding[/yellow]\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
