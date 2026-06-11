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

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

console = Console()
checks:  list[tuple] = []


def check(name: str, func) -> None:
    try:
        result = func()
        checks.append((name, "✓", "green", result or "OK"))
    except Exception as e:
        checks.append((name, "✗", "red", str(e)[:60]))


# ── 1. Dependencias Python ────────────────────────────────────

def verify_imports() -> str:
    # FIX [V1]: separar core obligatorias de opcionales
    core_required = [
        "fastapi", "uvicorn", "websockets", "pydantic",
        "sklearn", "networkx", "numpy",
        "loguru", "rich", "dotenv",
    ]
    llm_optional = ["google.genai", "anthropic", "openai"]
    data_optional = ["datasets", "pandas"]

    missing_core = []
    for imp in core_required:
        try:
            __import__(imp)
        except ImportError:
            missing_core.append(imp)

    if missing_core:
        raise ImportError(f"Faltan dependencias core: {', '.join(missing_core)}")

    missing_llm  = [i for i in llm_optional  if not _can_import(i)]
    missing_data = [i for i in data_optional if not _can_import(i)]

    parts = [f"{len(core_required)}/{len(core_required)} core OK"]
    if missing_llm:
        parts.append(f"LLM opcionales sin instalar: {', '.join(missing_llm)}")
    if missing_data:
        parts.append(f"Data opcionales: {', '.join(missing_data)} (solo para download_data.py)")
    return " | ".join(parts)


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


# ── 2. Módulos del proyecto ───────────────────────────────────

def verify_core_modules() -> str:
    modules = [
        "backend.core.graph",
        "backend.core.generator",
        "backend.core.evaluator",
        "backend.core.constraints",
        "backend.core.scorer",
        "backend.core.simulation",
        "backend.data.loader",
        "backend.data.input_manager",
        "backend.llm.client",
        "backend.llm.analyzer",
        "backend.llm.prompts",
    ]
    ok = sum(1 for m in modules if _can_import(m))
    if ok < len(modules):
        raise ImportError(f"Solo {ok}/{len(modules)} módulos encontrados")
    return f"{ok}/{len(modules)} módulos OK"


# ── 3. Archivos clave ─────────────────────────────────────────

def verify_files() -> str:
    files = [
        "backend/main_api.py",
        "backend/data/careers.json",
        "backend/data/input_manager.py",
        "main.py",
        "frontend/index.html",
        "frontend/js/websocket.js",
        ".env.example",
    ]
    root  = Path(__file__).parent
    found = sum(1 for f in files if (root / f).exists())
    if found < len(files):
        missing = [f for f in files if not (root / f).exists()]
        return f"{found}/{len(files)} archivos OK | Faltan: {', '.join(missing)}"
    return f"{found}/{len(files)} archivos OK"


# ── 4. Base de datos ──────────────────────────────────────────

def verify_database() -> str:
    from backend.data.input_manager import InputManager
    manager = InputManager()
    inputs  = manager.list_inputs()
    return f"SQLite operativa | {len(inputs)} configuraciones guardadas"


# ── 5. Configuración LLM ─────────────────────────────────────

def verify_llm_setup() -> str:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    # FIX [V2]: filtrar SOLO LLM_KEY_N, no otras variables LLM_*
    env_keys = [k for k in os.environ if k.startswith("LLM_KEY_")]
    if not env_keys:
        return "⚠️  Sin keys LLM (configura .env con LLM_KEY_1=proveedor:key)"
    return f"{len(env_keys)} key(s) configurada(s): {', '.join(sorted(env_keys))}"


# ── 6. Grafo de carreras ──────────────────────────────────────

def verify_graph() -> str:
    from backend.data.loader import load_career_graph
    from backend.core.graph import CareerGraph

    G         = load_career_graph()
    graph     = CareerGraph(G)
    nodes     = graph.all_node_ids()
    terminals = graph.terminal_nodes()

    # Verificar campos v2
    sample_scores = graph.score_trajectory(("junior_dev", "mid_dev", "senior_dev"))
    has_v2_fields = (
        "is_terminal_end" in sample_scores
        and "transition_probability_score" in sample_scores
    )

    return (
        f"{len(nodes)} nodos | {len(terminals)} terminales | "
        f"campos v2: {'✓' if has_v2_fields else '✗'}"
    )


# ── 7. Predictor ML ───────────────────────────────────────────

def verify_predictor() -> str:
    from backend.core.scorer import CareerOutcomePredictor
    p = CareerOutcomePredictor.load_or_train()
    top = max(p.feature_importances, key=p.feature_importances.get) if p.feature_importances else "N/A"
    return f"CV AUC={p.cv_score:.3f} | Top feature: {top}"


# ── 8. Domain graphs (opcional) ───────────────────────────────

def verify_domain_graphs() -> str:
    from backend.data.loader import list_available_domains
    domains = list_available_domains()
    if not domains:
        return "Sin domain graphs (ejecuta python backend/data/transform_data.py)"
    return f"{len(domains)} domain graphs disponibles"


# ────────────────────────────────────────────────────────────────

def main() -> int:
    console.print(Text("\n🔍 PathForge — Verificación de Instalación\n", style="bold cyan"))

    check("Dependencias Python",   verify_imports)
    check("Módulos del proyecto",  verify_core_modules)
    check("Archivos del proyecto", verify_files)
    check("Base de datos SQLite",  verify_database)
    check("Configuración LLM",     verify_llm_setup)
    check("Grafo de carreras",     verify_graph)
    check("Modelo ML (sklearn)",   verify_predictor)
    check("Domain graphs",         verify_domain_graphs)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Componente", style="cyan", min_width=25)
    table.add_column("Estado",     width=4)
    table.add_column("Detalle",    style="dim")

    failed = 0
    for name, status, color, details in checks:
        if status == "✗":
            failed += 1
        table.add_row(name, Text(status, style=color), details)

    console.print(table)

    if failed == 0:
        console.print("\n[bold green]✓ Todos los componentes verificados[/bold green]")
        console.print("[dim]Siguiente paso: python main.py server[/dim]\n")
        return 0
    else:
        console.print(f"\n[red]✗ {failed} componente(s) con error[/red]")
        console.print("[yellow]Revisa los errores y ejecuta: pip install -r requirements.txt[/yellow]\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
