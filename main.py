"""
main.py
-------
Punto de entrada de PathForge — CLI mejorada con comandos separados.

Comandos:
    python main.py train                 # Entrenar modelos de IA
    python main.py run --input preset_junior_conservative
    python main.py server                # Iniciar servidor FastAPI
    python main.py input create          # Crear nueva entrada
    python main.py input list            # Listar todas las entradas
    python main.py input load <id>       # Cargar y explorar entrada
"""

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from backend.core.constraints import ConstraintProfiles, Constraint
from backend.core.generator import GeneratorConfig, TrajectoryGenerator
from backend.core.graph import CareerGraph
from backend.core.scorer import CareerOutcomePredictor
from backend.data.loader import load_career_graph
from backend.data.input_manager import InputManager, UserInput, create_default_presets
from backend.llm.analyzer import TrajectoryAnalyzer

console = Console()

# ────────────────────────────────────────────────────────────────
# Configuración
# ────────────────────────────────────────────────────────────────

BANNER = r"""
████████████████████████████████████████████████████████████████████████████
█▌  PathForge · Career Universe Explorer  ▐█
████████████████████████████████████████████████████████████████████████████
"""

PROFILES = {
    "conservative": ConstraintProfiles.conservative,
    "ambitious":    ConstraintProfiles.ambitious,
    "balanced":     ConstraintProfiles.balanced,
    "fast_track":   ConstraintProfiles.fast_track,
}

PROFILE_DESCRIPTIONS = {
    "conservative": "🛡️  Bajo riesgo — estabilidad",
    "ambitious":    "📈 Salario máximo — crecimiento económico",
    "balanced":     "⚖️  Equilibrio — tiempo, riesgo, dinero",
    "fast_track":   "⚡ Máximo crecimiento en menor tiempo",
}


def _resolve_source_node(raw_source: str, graph: CareerGraph) -> str:
    """Acepta ID exacto o label human-friendly y devuelve el node_id real."""
    source = raw_source.strip()
    node_ids = set(graph.all_node_ids())
    if source in node_ids:
        return source

    normalized = source.lower().strip()
    for nid in graph.all_node_ids():
        label = str(graph.node_attrs(nid).get("label", "")).lower().strip()
        if normalized == label:
            return nid

    raise ValueError(
        f"Carrera inicial '{raw_source}' no existe. Usa un ID válido como: "
        f"{', '.join(sorted(graph.all_node_ids())[:6])}..."
    )


# ────────────────────────────────────────────────────────────────
# COMANDO 1: train — Entrenar modelos de IA
# ────────────────────────────────────────────────────────────────

def cmd_train():
    """Entrena los modelos de IA (sklearn + LLM)."""
    console.print(Panel(
        "[bold cyan]Entrenando modelos de IA...[/bold cyan]\n"
        "[dim]Esto incluye: predictor de resultados (sklearn), carga de grafo y configuración de LLM[/dim]",
        border_style="cyan", title="🎓 ENTRENAMIENTO"
    ))

    from backend.core.scorer import CareerOutcomePredictor
    from backend.data.loader import load_career_graph

    try:
        console.print("[yellow]» Cargando grafo profesional...[/yellow]")
        G = load_career_graph()
        console.print(f"[green]✓ Grafo cargado: {len(G.nodes)} nodos, {len(G.edges)} aristas[/green]")

        console.print("[yellow]» Entrenando predictor de resultados (sklearn)...[/yellow]")
        predictor = CareerOutcomePredictor.load_or_train()
        console.print(f"[green]✓ Predictor entrenado y guardado[/green]")

        from backend.llm.client import get_llm_client
        console.print("[yellow]» Validando clientes LLM...[/yellow]")
        client = get_llm_client()
        console.print(f"[green]✓ LLM listo: {client.active_provider}[/green]")

        console.print("\n[bold green]✓ Entrenamiento completado exitosamente[/bold green]")
        console.print("[dim]Los modelos están listos para ejecutar el programa.[/dim]")

    except Exception as e:
        console.print(f"[red]✗ Error durante entrenamiento: {e}[/red]", style="red")
        sys.exit(1)


# ────────────────────────────────────────────────────────────────
# COMANDO 2: run — Ejecutar exploración interactiva
# ────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace):
    """Ejecuta exploración interactiva usando un input guardado."""
    console.print(Panel("[bold cyan]Modo Exploración Interactiva[/bold cyan]", border_style="cyan"))

    manager = InputManager()

    # Opción 1: Usar un preset
    if args.input:
        user_input = manager.load_input(args.input)
        if not user_input:
            console.print(f"[red]✗ Input '{args.input}' no encontrado[/red]")
            _cmd_input_list()
            sys.exit(1)
        console.print(f"[green]✓ Cargado: {args.input}[/green]")
    else:
        # Opción 2: Crear uno nuevo interactivamente
        user_input = _interactive_input_setup()
        if not user_input:
            return

    _run_exploration(user_input)


def _interactive_input_setup() -> UserInput | None:
    """Crea una entrada interactiva."""
    console.print("\n[bold]Crear nueva entrada[/bold]")

    input_id = Prompt.ask("ID de entrada", default=f"custom_{Path.cwd().resolve().stem}")
    source = Prompt.ask("Carrera inicial", default="junior_dev")

    console.print("\nPerfiles disponibles:")
    for k, desc in PROFILE_DESCRIPTIONS.items():
        console.print(f"  [bold]{k}[/bold] — {desc}")
    profile = Prompt.ask("Perfil", default="balanced", choices=list(PROFILES.keys()))

    max_years = int(Prompt.ask("Años máximos", default="12"))
    max_risk = float(Prompt.ask("Riesgo máximo (0-1)", default="0.6"))
    user_profile = Prompt.ask("Describe tu perfil", default="profesional de tecnología")

    return UserInput(
        id=input_id,
        source_career=source,
        profile=profile,
        max_years=max_years,
        max_risk=max_risk,
        user_profile_description=user_profile,
    )


def _run_exploration(user_input: UserInput):
    """Ejecuta la exploración con los parámetros dados."""
    console.print(Panel(
        f"[bold cyan]Exploración iniciada[/bold cyan]\n"
        f"Carrera: {user_input.source_career} | Perfil: {user_input.profile}\n"
        f"Años: {user_input.max_years} | Riesgo máx: {user_input.max_risk}",
        border_style="cyan"
    ))

    try:
        # Cargar grafo y crear generador
        graph = CareerGraph(load_career_graph())
        source_node = _resolve_source_node(user_input.source_career, graph)
        profile_fn = PROFILES[user_input.profile]
        constraints = profile_fn(user_input.max_years) if user_input.profile == "balanced" else profile_fn()

        config = GeneratorConfig(
            beam_width=user_input.beam_width,
            max_depth=user_input.max_depth,
            top_k_results=user_input.top_k,
        )

        console.print(f"\n[yellow]» Generando trayectorias...[/yellow]")
        generator = TrajectoryGenerator(graph, config)
        results = generator.generate(source_node, constraints)

        _print_results_table(results)

        # Análisis con IA
        if results and Prompt.ask("\n¿Analizar con IA?", choices=["y", "n"], default="y") == "y":
            criterion = Prompt.ask("Objetivo", default="crecimiento general balanceado")
            analyzer = TrajectoryAnalyzer(user_profile=user_input.user_profile_description)

            console.print("[yellow]» Obteniendo análisis de IA...[/yellow]")
            analysis = analyzer.rank_by(results, criterion)
            console.print(Panel(analysis.content, border_style="magenta", title="📊 Análisis de IA"))

    except Exception as e:
        console.print(f"[red]✗ Error en exploración: {e}[/red]")
        sys.exit(1)


def _print_results_table(results):
    """Imprime tabla de resultados."""
    if not results:
        console.print("[yellow]No hay trayectorias que mostrar[/yellow]")
        return

    table = Table(title="🎯 Trayectorias Generadas", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Trayectoria", style="cyan", min_width=40)
    table.add_column("💰 Salario", justify="right", style="green")
    table.add_column("📈 Crec.", justify="right", style="yellow")
    table.add_column("⏱ Años", justify="right")
    table.add_column("⚠️ Riesgo", justify="right")
    table.add_column("😊 Sat.", justify="right")
    table.add_column("🏆 Rk", justify="center")

    for i, et in enumerate(results[:15], 1):
        s = et.scores
        path = " → ".join(et.trajectory.nodes)
        badge = "⭐" * max(0, 3 - et.pareto_rank) if et.pareto_rank < 3 else "·"

        table.add_row(
            str(i),
            path[:35] + "..." if len(path) > 35 else path,
            f"${s.get('final_salary', 0):,.0f}",
            f"{s.get('salary_growth', 0):.0%}",
            f"{s.get('total_years', 0):.0f}",
            f"{s.get('avg_risk', 0):.0%}",
            f"{s.get('avg_satisfaction', 0):.0%}",
            badge,
        )

    console.print(table)


# ────────────────────────────────────────────────────────────────
# COMANDO 3: server — Iniciar servidor FastAPI
# ────────────────────────────────────────────────────────────────

def cmd_server(args: argparse.Namespace):
    """Inicia el servidor FastAPI con frontend."""
    console.print(Panel(
        "[bold cyan]Iniciando servidor FastAPI[/bold cyan]\n"
        "[dim]Backend en http://localhost:8000[/dim]\n"
        "[dim]Frontend en http://localhost:8000[/dim]",
        border_style="cyan", title="🚀 SERVIDOR"
    ))

    host = args.host or "0.0.0.0"
    port = args.port or 8000

    try:
        import uvicorn
        uvicorn.run(
            "backend.main_api:app",
            host=host,
            port=port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Servidor detenido[/yellow]")


# ────────────────────────────────────────────────────────────────
# COMANDO 4: input — Gestionar entradas de usuario
# ────────────────────────────────────────────────────────────────

def _cmd_input_create():
    """Crear nueva entrada."""
    manager = InputManager()

    inp = _interactive_input_setup()
    if inp:
        manager.save_input(inp)
        console.print(f"[green]✓ Entrada '{inp.id}' creada[/green]")


def _cmd_input_list():
    """Listar todas las entradas."""
    manager = InputManager()
    inputs = manager.list_inputs()

    if not inputs:
        console.print("[yellow]No hay entradas guardadas[/yellow]")
        return

    table = Table(title="📝 Entradas Guardadas", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Carrera", style="green")
    table.add_column("Perfil", style="yellow")
    table.add_column("Años", justify="right")
    table.add_column("Riesgo", justify="right")
    table.add_column("Actualizado", style="dim")

    for inp in inputs:
        table.add_row(
            inp.id,
            inp.source_career,
            inp.profile,
            str(inp.max_years),
            f"{inp.max_risk:.2f}",
            inp.updated_at[:10] if inp.updated_at else "—",
        )

    console.print(table)


def _cmd_input_load(input_id: str):
    """Cargar y ejecutar una entrada."""
    manager = InputManager()
    inp = manager.load_input(input_id)

    if not inp:
        console.print(f"[red]✗ Entrada '{input_id}' no encontrada[/red]")
        _cmd_input_list()
        return

    console.print(f"[green]✓ Cargada: {input_id}[/green]")
    _run_exploration(inp)


def _cmd_input_delete(input_id: str):
    """Eliminar entrada."""
    manager = InputManager()
    if manager.load_input(input_id):
        manager.delete_input(input_id)
        console.print(f"[green]✓ Entrada '{input_id}' eliminada[/green]")
    else:
        console.print(f"[red]✗ Entrada '{input_id}' no encontrada[/red]")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PathForge — Career Universe Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py train                          # Entrenar IA
  python main.py run --input preset_junior_conservative  # Usar preset
  python main.py server                         # Iniciar servidor
  python main.py input list                     # Listar entradas
  python main.py input create                   # Crear entrada nueva
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # train
    subparsers.add_parser("train", help="Entrenar modelos de IA")

    # run
    run_parser = subparsers.add_parser("run", help="Ejecutar exploración")
    run_parser.add_argument("--input", "-i", help="ID de entrada guardada")

    # server
    server_parser = subparsers.add_parser("server", help="Iniciar servidor FastAPI")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=8000)
    server_parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")

    # input
    input_parser = subparsers.add_parser("input", help="Gestionar entradas")
    input_subparsers = input_parser.add_subparsers(dest="input_command", help="Subcomando")
    input_subparsers.add_parser("list", help="Listar todas las entradas")
    input_subparsers.add_parser("create", help="Crear nueva entrada")
    load_parser = input_subparsers.add_parser("load", help="Cargar y ejecutar entrada")
    load_parser.add_argument("id", help="ID de entrada")
    delete_parser = input_subparsers.add_parser("delete", help="Eliminar entrada")
    delete_parser.add_argument("id", help="ID de entrada")

    args = parser.parse_args()

    # Mostrar banner
    console.print(Text(BANNER, style="bold cyan"))

    # Crear presets por defecto si no existen
    manager = InputManager()
    create_default_presets(manager)

    # Ejecutar comando
    if args.command == "train":
        cmd_train()
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "server":
        cmd_server(args)
    elif args.command == "input":
        if args.input_command == "list":
            _cmd_input_list()
        elif args.input_command == "create":
            _cmd_input_create()
        elif args.input_command == "load":
            _cmd_input_load(args.id)
        elif args.input_command == "delete":
            _cmd_input_delete(args.id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
