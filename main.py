"""
main.py
-------
Punto de entrada de PathForge — CLI con comandos separados.

Comandos:
    python main.py train                                   # Entrenar modelos de IA
    python main.py run --input preset_junior_conservative  # Explorar con preset
    python main.py server                                  # Servidor con careers.json (default)
    python main.py server --domain assembly                # Servidor con dominio real específico
    python main.py server --empty                          # Servidor sin grafo (lo rellenas en UI)
    python main.py input list                              # Listar entradas guardadas
    python main.py input create                            # Crear nueva entrada
    python main.py input load <id>                         # Cargar y explorar entrada
    python main.py input delete <id>                       # Eliminar entrada
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
from backend.data.loader import load_career_graph, list_available_domains
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
    source   = raw_source.strip()
    node_ids = set(graph.all_node_ids())
    if source in node_ids:
        return source
    normalized = source.lower().strip()
    for nid in graph.all_node_ids():
        label = str(graph.node_attrs(nid).get("label", "")).lower().strip()
        if normalized == label:
            return nid
    available = sorted(
        f"{nid} ({graph.node_attrs(nid).get('label', nid)})"
        for nid in graph.all_node_ids()
    )
    raise ValueError(
        f"Carrera inicial '{raw_source}' no existe.\n"
        f"Opciones:\n  " + "\n  ".join(available)
    )


# ────────────────────────────────────────────────────────────────
# COMANDO 1: train
# ────────────────────────────────────────────────────────────────

def cmd_train():
    console.print(Panel(
        "[bold cyan]Entrenando modelos de IA...[/bold cyan]\n"
        "[dim]Predictor sklearn + validación de claves LLM[/dim]",
        border_style="cyan", title="🎓 ENTRENAMIENTO"
    ))
    try:
        console.print("[yellow]» Cargando grafo profesional...[/yellow]")
        G = load_career_graph()
        console.print(f"[green]✓ Grafo: {len(G.nodes)} nodos, {len(G.edges)} aristas[/green]")

        console.print("[yellow]» Entrenando predictor ML (sklearn)...[/yellow]")
        predictor = CareerOutcomePredictor.load_or_train()
        console.print(f"[green]✓ Predictor | CV AUC={predictor.cv_score:.3f}[/green]")

        console.print("[yellow]» Validando LLM...[/yellow]")
        try:
            from backend.llm.client import get_llm_client
            client = get_llm_client()
            console.print(f"[green]✓ LLM: {client.active_provider} ({client.key_count} key(s))[/green]")
        except EnvironmentError as e:
            console.print(f"[yellow]⚠ LLM no configurado: {e}[/yellow]")

        console.print("\n[bold green]✓ Listo[/bold green] — ejecuta: python main.py server")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        sys.exit(1)


# ────────────────────────────────────────────────────────────────
# COMANDO 2: run
# ────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace):
    console.print(Panel("[bold cyan]Modo Exploración Interactiva[/bold cyan]", border_style="cyan"))
    manager = InputManager()
    create_default_presets(manager)

    if args.input:
        user_input = manager.load_input(args.input)
        if not user_input:
            console.print(f"[red]✗ Input '{args.input}' no encontrado[/red]")
            _cmd_input_list()
            sys.exit(1)
    else:
        user_input = _interactive_input_setup()
        if not user_input:
            return

    _run_exploration(user_input)


def _interactive_input_setup() -> UserInput | None:
    console.print("\n[bold]Nueva configuración[/bold]")
    input_id  = Prompt.ask("ID", default="custom_session")
    source    = Prompt.ask("Carrera inicial", default="junior_dev")
    for k, desc in PROFILE_DESCRIPTIONS.items():
        console.print(f"  [bold]{k}[/bold] — {desc}")
    profile   = Prompt.ask("Perfil", default="balanced", choices=list(PROFILES.keys()))
    max_years = int(Prompt.ask("Años máximos", default="12"))
    max_risk  = float(Prompt.ask("Riesgo máximo (0–1)", default="0.6"))
    user_prof = Prompt.ask("Describe tu perfil", default="profesional de tecnología")
    return UserInput(id=input_id, source_career=source, profile=profile,
                     max_years=max_years, max_risk=max_risk, user_profile_description=user_prof)


def _run_exploration(user_input: UserInput) -> None:
    try:
        graph       = CareerGraph(load_career_graph())
        source_node = _resolve_source_node(user_input.source_career, graph)
        profile_fn  = PROFILES[user_input.profile]
        constraints = profile_fn(user_input.max_years) if user_input.profile == "balanced" else profile_fn()
        config      = GeneratorConfig(beam_width=user_input.beam_width,
                                      max_depth=user_input.max_depth,
                                      top_k_results=user_input.top_k)
        generator   = TrajectoryGenerator(graph, config)
        console.print("\n[yellow]» Generando trayectorias...[/yellow]")
        results     = generator.generate(source_node, constraints)
        _print_results_table(results)

        if not results:
            console.print("[yellow]Sin trayectorias. Prueba un perfil más permisivo.[/yellow]")
            return

        if Prompt.ask("\n¿Analizar con IA?", choices=["y", "n"], default="y") == "y":
            criterion = Prompt.ask("Objetivo", default="crecimiento equilibrado a largo plazo")
            analyzer  = TrajectoryAnalyzer(user_profile=user_input.user_profile_description)
            analysis  = analyzer.rank_by(results, criterion)
            console.print(Panel(analysis.content, border_style="magenta",
                                title=f"📊 [{analysis.provider_used}]"))
    except Exception as e:
        console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)


def _print_results_table(results: list) -> None:
    if not results:
        return
    table = Table(title="🎯 Trayectorias", show_header=True, header_style="bold magenta")
    table.add_column("#", width=3); table.add_column("Trayectoria", style="cyan", min_width=38)
    table.add_column("💰", justify="right", style="green"); table.add_column("📈", justify="right")
    table.add_column("⏱", justify="right"); table.add_column("⚠️", justify="right")
    table.add_column("🏆", justify="center")
    for i, et in enumerate(results[:15], 1):
        s    = et.scores
        path = " → ".join(et.trajectory.nodes)
        badge = "⭐" * max(0, 3 - et.pareto_rank) if et.pareto_rank < 3 else "·"
        table.add_row(str(i), path[:36] + "..." if len(path) > 36 else path,
                      f"${s.get('final_salary',0):,.0f}", f"{s.get('salary_growth',0):.0%}",
                      f"{s.get('total_years',0):.0f}", f"{s.get('avg_risk',0):.0%}", badge)
    console.print(table)


# ────────────────────────────────────────────────────────────────
# COMANDO 3: server  ← NUEVO: --domain y --empty
# ────────────────────────────────────────────────────────────────

def cmd_server(args: argparse.Namespace) -> None:
    """
    Inicia el servidor FastAPI.

    Modos:
      python main.py server                   → carga careers.json (default)
      python main.py server --domain assembly → preselecciona 'assembly' en la UI
      python main.py server --empty           → arranca con grafo vacío (rellenas en UI)
    """
    # ── Validar --domain si se pasó ─────────────────────────────
    startup_domain = None

    if getattr(args, 'empty', False):
        startup_domain = "__empty__"
        console.print(Panel(
            "[bold cyan]Modo Empty — grafo en blanco[/bold cyan]\n"
            "[dim]La interfaz arrancará sin ningún grafo cargado.\n"
            "Añade nodos y wormholes manualmente en la vista Setup.[/dim]",
            border_style="yellow", title="🚀 SERVIDOR — EMPTY MODE"
        ))

    elif getattr(args, 'domain', None):
        domain_id = args.domain.strip()
        # Validar que el dominio existe
        available = {d["id"]: d for d in list_available_domains()}
        if domain_id not in available:
            console.print(f"[red]✗ Dominio '{domain_id}' no encontrado.[/red]")
            if available:
                console.print("\n[yellow]Dominios disponibles:[/yellow]")
                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("ID", style="cyan"); table.add_column("Sector")
                table.add_column("Nodos", justify="right"); table.add_column("Aristas", justify="right")
                for d in sorted(available.values(), key=lambda x: x["id"])[:30]:
                    table.add_row(d["id"], d.get("broad_sector","—"),
                                  str(d["nodes"]), str(d["edges"]))
                console.print(table)
                console.print("\n[dim]Ejemplo: python main.py server --domain software_development[/dim]")
            else:
                console.print("[dim]No hay domain graphs. Ejecuta: python backend/data/transform_data.py[/dim]")
            sys.exit(1)

        startup_domain = domain_id
        d = available[domain_id]
        console.print(Panel(
            f"[bold cyan]Domain Graph: [yellow]{domain_id}[/yellow][/bold cyan]\n"
            f"Sector: {d.get('broad_sector','—')} | "
            f"{d['nodes']} nodos | {d['edges']} aristas\n"
            f"[dim]La UI arrancará con este dominio preseleccionado.[/dim]",
            border_style="cyan", title="🚀 SERVIDOR — DOMAIN MODE"
        ))

    else:
        console.print(Panel(
            "[bold cyan]Servidor PathForge[/bold cyan]\n"
            f"[dim]http://{args.host}:{args.port}  |  careers.json (default)[/dim]\n"
            "[dim]Otras opciones:\n"
            "  --domain <id>   → precargar un dominio real\n"
            "  --empty         → empezar con grafo vacío[/dim]",
            border_style="cyan", title="🚀 SERVIDOR"
        ))

    # ── Inyectar config en main_api antes de arrancar ───────────
    from backend.main_api import set_startup_domain
    set_startup_domain(startup_domain)

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
        console.print(f"[green]✓ '{inp.id}' guardada[/green]")


def _cmd_input_list() -> None:
    manager = InputManager()
    inputs  = manager.list_inputs()
    if not inputs:
        console.print("[yellow]Sin configuraciones. Crea una con: python main.py input create[/yellow]")
        return
    table = Table(title="📝 Configuraciones", show_header=True)
    table.add_column("ID", style="cyan"); table.add_column("Carrera", style="green")
    table.add_column("Dominio", style="blue"); table.add_column("Perfil", style="yellow")
    table.add_column("Años", justify="right"); table.add_column("Riesgo", justify="right")
    for inp in inputs:
        table.add_row(inp.id, inp.source_career, inp.domain_id or "default",
                      inp.profile, str(inp.max_years), f"{inp.max_risk:.2f}")
    console.print(table)


def _cmd_input_load(input_id: str) -> None:
    manager = InputManager()
    inp     = manager.load_input(input_id)
    if not inp:
        console.print(f"[red]✗ '{input_id}' no encontrada[/red]")
        _cmd_input_list()
        return
    _run_exploration(inp)


def _cmd_input_delete(input_id: str) -> None:
    manager = InputManager()
    if manager.load_input(input_id):
        manager.delete_input(input_id)
        console.print(f"[green]✓ '{input_id}' eliminada[/green]")
    else:
        console.print(f"[red]✗ '{input_id}' no encontrada[/red]")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PathForge — Career Universe Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py train                                    # Entrenar IA
  python main.py server                                   # Servidor con careers.json
  python main.py server --domain assembly                 # Precargar dominio 'assembly'
  python main.py server --domain software_development     # Precargar software
  python main.py server --empty                           # Empezar con grafo vacío
  python main.py server --reload                          # Auto-reload (desarrollo)
  python main.py run --input preset_junior_conservative   # Exploración CLI
  python main.py input list                               # Ver configuraciones
  python main.py input create                             # Nueva configuración
  python main.py input load preset_data_scientist         # Cargar y explorar
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # train
    subparsers.add_parser("train", help="Entrenar modelos de IA")

    # run
    run_p = subparsers.add_parser("run", help="Exploración interactiva CLI")
    run_p.add_argument("--input", "-i", help="ID de configuración guardada")

    # server  ← los dos flags nuevos
    srv_p = subparsers.add_parser("server", help="Iniciar servidor FastAPI + frontend")
    srv_p.add_argument("--host",   default="0.0.0.0")
    srv_p.add_argument("--port",   type=int, default=8000)
    srv_p.add_argument("--reload", action="store_true", help="Auto-reload (desarrollo)")

    # Grupo mutuamente excluyente: --domain XOR --empty
    domain_group = srv_p.add_mutually_exclusive_group()
    domain_group.add_argument(
        "--domain", metavar="DOMAIN_ID",
        help="Precargar un domain graph específico en la UI (ej: assembly, software_development)"
    )
    domain_group.add_argument(
        "--empty", action="store_true",
        help="Arrancar con grafo vacío — rellena los nodos tú mismo en la interfaz"
    )

    # input
    inp_p  = subparsers.add_parser("input", help="Gestionar configuraciones")
    inp_sp = inp_p.add_subparsers(dest="input_command")
    inp_sp.add_parser("list")
    inp_sp.add_parser("create")
    lp = inp_sp.add_parser("load");   lp.add_argument("id")
    dp = inp_sp.add_parser("delete"); dp.add_argument("id")

    args = parser.parse_args()
    console.print(Text(BANNER, style="bold cyan"))

    if   args.command == "train":  cmd_train()
    elif args.command == "run":    cmd_run(args)
    elif args.command == "server": cmd_server(args)
    elif args.command == "input":
        if   args.input_command == "list":   _cmd_input_list()
        elif args.input_command == "create": _cmd_input_create()
        elif args.input_command == "load":   _cmd_input_load(args.id)
        elif args.input_command == "delete": _cmd_input_delete(args.id)
        else: inp_p.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
