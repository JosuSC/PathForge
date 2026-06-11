"""
main.py
-------
Punto de entrada de PathForge — CLI con comandos separados.

Comandos:
    python main.py train                              # Entrenar modelos de IA
    python main.py run --input preset_junior_conservative
    python main.py server                             # Iniciar servidor FastAPI
    python main.py input list                         # Listar entradas guardadas
    python main.py input create                       # Crear nueva entrada
    python main.py input load <id>                    # Cargar y explorar entrada
    python main.py input delete <id>                  # Eliminar entrada
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from backend.core.constraints import ConstraintProfiles
from backend.core.generator import GeneratorConfig, TrajectoryGenerator
from backend.core.graph import CareerGraph
from backend.core.scorer import CareerOutcomePredictor
from backend.data.loader import load_career_graph
from backend.data.input_manager import InputManager, UserInput, create_default_presets
from backend.llm.analyzer import TrajectoryAnalyzer

console = Console()

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
    source   = raw_source.strip()
    node_ids = set(graph.all_node_ids())
    if source in node_ids:
        return source

    normalized = source.lower().strip()
    for nid in graph.all_node_ids():
        label = str(graph.node_attrs(nid).get("label", "")).lower().strip()
        if normalized == label:
            return nid

    # FIX [M3]: mostrar labels disponibles en el error para ayudar al usuario
    available = sorted(
        f"{nid} ({graph.node_attrs(nid).get('label', nid)})"
        for nid in graph.all_node_ids()
    )
    raise ValueError(
        f"Carrera inicial '{raw_source}' no existe.\n"
        f"Opciones disponibles:\n  " + "\n  ".join(available)
    )


# ────────────────────────────────────────────────────────────────
# COMANDO 1: train
# ────────────────────────────────────────────────────────────────

def cmd_train():
    """Entrena el predictor ML y valida la configuración LLM."""
    console.print(Panel(
        "[bold cyan]Entrenando modelos de IA...[/bold cyan]\n"
        "[dim]Predictor sklearn + validación de claves LLM[/dim]",
        border_style="cyan", title="🎓 ENTRENAMIENTO"
    ))

    try:
        console.print("[yellow]» Cargando grafo profesional...[/yellow]")
        G = load_career_graph()
        console.print(f"[green]✓ Grafo: {len(G.nodes)} nodos, {len(G.edges)} aristas[/green]")

        console.print("[yellow]» Entrenando predictor de resultados (sklearn)...[/yellow]")
        predictor = CareerOutcomePredictor.load_or_train()
        console.print(
            f"[green]✓ Predictor entrenado | CV AUC={predictor.cv_score:.3f}[/green]"
        )
        # FIX [M2]: comentario explicativo — el predictor se conecta automáticamente
        # en main_api.py via CareerGraph(raw_graph, outcome_predictor=predictor)
        console.print(
            "[dim]  El predictor se conecta al grafo automáticamente al iniciar el servidor.[/dim]"
        )

        console.print("[yellow]» Validando clientes LLM...[/yellow]")
        try:
            from backend.llm.client import get_llm_client
            client = get_llm_client()
            console.print(f"[green]✓ LLM listo: {client.active_provider} "
                          f"({client.key_count} key(s))[/green]")
        except EnvironmentError as e:
            console.print(f"[yellow]⚠ LLM no configurado: {e}[/yellow]")
            console.print("[dim]  Configura LLM_KEY_1=proveedor:api_key en .env[/dim]")

        console.print("\n[bold green]✓ Entrenamiento completado[/bold green]")
        console.print("[dim]Ejecuta: python main.py server[/dim]")

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        sys.exit(1)


# ────────────────────────────────────────────────────────────────
# COMANDO 2: run
# ────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace):
    """Ejecuta exploración interactiva usando un input guardado o creando uno nuevo."""
    console.print(Panel("[bold cyan]Modo Exploración Interactiva[/bold cyan]",
                        border_style="cyan"))

    manager = InputManager()
    # FIX [M4]: crear presets solo en comandos que los necesitan
    create_default_presets(manager)

    if args.input:
        user_input = manager.load_input(args.input)
        if not user_input:
            console.print(f"[red]✗ Input '{args.input}' no encontrado[/red]")
            _cmd_input_list()
            sys.exit(1)
        console.print(f"[green]✓ Cargado: {args.input}[/green]")
    else:
        user_input = _interactive_input_setup()
        if not user_input:
            return

    _run_exploration(user_input)


def _interactive_input_setup() -> UserInput | None:
    """Crea una entrada interactiva pidiendo datos al usuario."""
    console.print("\n[bold]Crear nueva configuración[/bold]")

    input_id     = Prompt.ask("ID de configuración", default="custom_session")
    source       = Prompt.ask("Carrera inicial (ID o nombre)", default="junior_dev")
    console.print("\nPerfiles disponibles:")
    for k, desc in PROFILE_DESCRIPTIONS.items():
        console.print(f"  [bold]{k}[/bold] — {desc}")
    profile      = Prompt.ask("Perfil", default="balanced", choices=list(PROFILES.keys()))
    max_years    = int(Prompt.ask("Años máximos", default="12"))
    max_risk     = float(Prompt.ask("Riesgo máximo (0.0–1.0)", default="0.6"))
    user_profile = Prompt.ask("Describe tu perfil", default="profesional de tecnología")

    return UserInput(
        id=input_id,
        source_career=source,
        profile=profile,
        max_years=max_years,
        max_risk=max_risk,
        user_profile_description=user_profile,
    )


def _run_exploration(user_input: UserInput) -> None:
    """Ejecuta la exploración con los parámetros dados."""
    console.print(Panel(
        f"[bold cyan]Exploración iniciada[/bold cyan]\n"
        f"Carrera: [yellow]{user_input.source_career}[/yellow] | "
        f"Perfil: [yellow]{user_input.profile}[/yellow]\n"
        f"Años máx: {user_input.max_years} | Riesgo máx: {user_input.max_risk}",
        border_style="cyan"
    ))

    try:
        graph       = CareerGraph(load_career_graph())
        source_node = _resolve_source_node(user_input.source_career, graph)
        profile_fn  = PROFILES[user_input.profile]

        # FIX [M1]: lógica clara y correcta para todos los perfiles
        if user_input.profile == "balanced":
            constraints = profile_fn(user_input.max_years)
        else:
            constraints = profile_fn()

        config    = GeneratorConfig(
            beam_width=user_input.beam_width,
            max_depth=user_input.max_depth,
            top_k_results=user_input.top_k,
        )
        generator = TrajectoryGenerator(graph, config)

        console.print("\n[yellow]» Generando trayectorias...[/yellow]")
        results = generator.generate(source_node, constraints)
        _print_results_table(results)

        if not results:
            console.print("[yellow]No se encontraron trayectorias con las restricciones actuales.[/yellow]")
            console.print("[dim]Prueba con un perfil más permisivo (ambitious) o más años.[/dim]")
            return

        if Prompt.ask("\n¿Analizar con IA?", choices=["y", "n"], default="y") == "y":
            criterion = Prompt.ask(
                "Describe tu objetivo", default="crecimiento equilibrado a largo plazo"
            )
            analyzer = TrajectoryAnalyzer(user_profile=user_input.user_profile_description)
            console.print("[yellow]» Obteniendo análisis de IA...[/yellow]")
            analysis = analyzer.rank_by(results, criterion)
            console.print(Panel(
                analysis.content,
                border_style="magenta",
                title=f"📊 Análisis — [{analysis.provider_used}]"
            ))

    except Exception as e:
        console.print(f"[red]✗ Error en exploración: {e}[/red]")
        sys.exit(1)


def _print_results_table(results: list) -> None:
    """Imprime tabla de resultados ordenada por Pareto rank."""
    if not results:
        return

    table = Table(
        title="🎯 Trayectorias Generadas",
        show_header=True, header_style="bold magenta"
    )
    table.add_column("#",       style="dim",    width=3)
    table.add_column("Trayectoria", style="cyan", min_width=40)
    table.add_column("💰 Salario",  justify="right", style="green")
    table.add_column("📈 Crec.",    justify="right", style="yellow")
    table.add_column("⏱ Años",     justify="right")
    table.add_column("⚠️ Riesgo",  justify="right")
    table.add_column("😊 Sat.",     justify="right")
    table.add_column("🏆 Rk",      justify="center")

    for i, et in enumerate(results[:15], 1):
        s     = et.scores
        path  = " → ".join(et.trajectory.nodes)
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
    console.print(
        f"[dim]Mostrando {min(15, len(results))} de {len(results)} trayectorias. "
        f"⭐ = Pareto rank 0 (óptimo)[/dim]"
    )


# ────────────────────────────────────────────────────────────────
# COMANDO 3: server
# ────────────────────────────────────────────────────────────────

def cmd_server(args: argparse.Namespace) -> None:
    """Inicia el servidor FastAPI con frontend."""
    console.print(Panel(
        "[bold cyan]Iniciando servidor PathForge[/bold cyan]\n"
        f"[dim]Backend + Frontend → http://{args.host}:{args.port}[/dim]\n"
        "[dim]API docs            → http://localhost:{args.port}/docs[/dim]",
        border_style="cyan", title="🚀 SERVIDOR"
    ))

    try:
        import uvicorn
        uvicorn.run(
            "backend.main_api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Servidor detenido[/yellow]")


# ────────────────────────────────────────────────────────────────
# COMANDO 4: input
# ────────────────────────────────────────────────────────────────

def _cmd_input_create() -> None:
    manager = InputManager()
    create_default_presets(manager)
    inp = _interactive_input_setup()
    if inp:
        manager.save_input(inp)
        console.print(f"[green]✓ Configuración '{inp.id}' guardada[/green]")


def _cmd_input_list() -> None:
    manager = InputManager()
    inputs  = manager.list_inputs()

    if not inputs:
        console.print("[yellow]No hay configuraciones guardadas.[/yellow]")
        console.print("[dim]Crea una con: python main.py input create[/dim]")
        return

    table = Table(title="📝 Configuraciones Guardadas", show_header=True)
    table.add_column("ID",       style="cyan")
    table.add_column("Carrera",  style="green")
    table.add_column("Dominio",  style="blue")
    table.add_column("Perfil",   style="yellow")
    table.add_column("Años",     justify="right")
    table.add_column("Riesgo",   justify="right")
    table.add_column("Actualizado", style="dim")

    for inp in inputs:
        table.add_row(
            inp.id,
            inp.source_career,
            inp.domain_id or "default",
            inp.profile,
            str(inp.max_years),
            f"{inp.max_risk:.2f}",
            inp.updated_at[:10] if inp.updated_at else "—",
        )

    console.print(table)


def _cmd_input_load(input_id: str) -> None:
    manager = InputManager()
    inp     = manager.load_input(input_id)
    if not inp:
        console.print(f"[red]✗ Configuración '{input_id}' no encontrada[/red]")
        _cmd_input_list()
        return
    console.print(f"[green]✓ Cargada: {input_id}[/green]")
    _run_exploration(inp)


def _cmd_input_delete(input_id: str) -> None:
    manager = InputManager()
    if manager.load_input(input_id):
        manager.delete_input(input_id)
        console.print(f"[green]✓ Configuración '{input_id}' eliminada[/green]")
    else:
        console.print(f"[red]✗ Configuración '{input_id}' no encontrada[/red]")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PathForge — Career Universe Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py train                                   # Entrenar IA
  python main.py run --input preset_junior_conservative  # Usar preset
  python main.py run                                     # Configuración interactiva
  python main.py server                                  # Iniciar servidor web
  python main.py server --reload                         # Servidor con auto-reload
  python main.py input list                              # Ver configuraciones guardadas
  python main.py input create                            # Nueva configuración
  python main.py input load preset_data_scientist        # Cargar y explorar
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # train
    subparsers.add_parser("train", help="Entrenar modelos de IA")

    # run
    run_p = subparsers.add_parser("run", help="Ejecutar exploración interactiva")
    run_p.add_argument("--input", "-i", help="ID de configuración guardada")

    # server
    srv_p = subparsers.add_parser("server", help="Iniciar servidor FastAPI + frontend")
    srv_p.add_argument("--host",   default="0.0.0.0")
    srv_p.add_argument("--port",   type=int, default=8000)
    srv_p.add_argument("--reload", action="store_true", help="Auto-reload al cambiar código")

    # input
    inp_p  = subparsers.add_parser("input", help="Gestionar configuraciones de usuario")
    inp_sp = inp_p.add_subparsers(dest="input_command", help="Subcomando")
    inp_sp.add_parser("list",   help="Listar todas las configuraciones")
    inp_sp.add_parser("create", help="Crear nueva configuración")
    lp = inp_sp.add_parser("load",   help="Cargar y ejecutar configuración")
    lp.add_argument("id", help="ID de configuración")
    dp = inp_sp.add_parser("delete", help="Eliminar configuración")
    dp.add_argument("id", help="ID de configuración")

    args = parser.parse_args()

    console.print(Text(BANNER, style="bold cyan"))

    # FIX [M4]: create_default_presets SOLO en comandos que los necesitan (run, input)
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
            inp_p.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
