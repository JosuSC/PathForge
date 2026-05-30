"""
main.py
-------
Punto de entrada de PathForge.

Ofrece una interfaz CLI interactiva para explorar trayectorias
profesionales, con output visual usando Rich.

Uso:
    python main.py                        # modo interactivo
    python main.py --source junior_dev    # inicio directo
    python main.py --source junior_dev --profile conservative
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from core.constraints import ConstraintProfiles, Constraint
from core.generator import GeneratorConfig, TrajectoryGenerator
from core.graph import CareerGraph
from data.loader import load_career_graph
from llm.analyzer import TrajectoryAnalyzer

console = Console()

# ---------------------------------------------------------------------------
# Constantes de presentación
# ---------------------------------------------------------------------------

BANNER = """
██████╗  █████╗ ████████╗██╗  ██╗███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔══██╗██╔══██╗╚══██╔══╝██║  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
██████╔╝███████║   ██║   ███████║█████╗  ██║   ██║██████╔╝██║  ███╗█████╗
██╔═══╝ ██╔══██║   ██║   ██╔══██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
██║     ██║  ██║   ██║   ██║  ██║██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
"""

PROFILES = {
    "conservative": ConstraintProfiles.conservative,
    "ambitious":    ConstraintProfiles.ambitious,
    "balanced":     ConstraintProfiles.balanced,
    "fast_track":   ConstraintProfiles.fast_track,
}

PROFILE_DESCRIPTIONS = {
    "conservative": "Bajo riesgo, baja dificultad — ideal si quieres estabilidad",
    "ambitious":    "Salario final alto — para quienes priorizan crecer económicamente",
    "balanced":     "Equilibrio entre tiempo, riesgo y salario (recomendado)",
    "fast_track":   "Máximo crecimiento en el menor tiempo posible",
}


# ---------------------------------------------------------------------------
# Helpers de visualización
# ---------------------------------------------------------------------------

def print_banner() -> None:
    console.print(Text(BANNER, style="bold cyan"))
    console.print(
        Panel(
            "[bold white]Explorador inteligente de trayectorias profesionales[/bold white]\n"
            "[dim]Algoritmo: Beam Search + Pareto Optimality + Gemini LLM[/dim]",
            border_style="cyan",
        )
    )


def print_trajectories_table(results: list) -> None:
    """Muestra las trayectorias generadas como tabla Rich."""
    table = Table(
        title="🎯 Trayectorias Generadas",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        show_lines=True,
    )

    table.add_column("#",           style="dim", width=3)
    table.add_column("Trayectoria", style="cyan", min_width=40)
    table.add_column("💰 Salario Final", justify="right", style="green")
    table.add_column("📈 Crecimiento",   justify="right", style="yellow")
    table.add_column("⏱  Años",          justify="right")
    table.add_column("⚠️  Riesgo",        justify="right")
    table.add_column("😊 Satisfacción",  justify="right")
    table.add_column("🏆 Pareto",        justify="center")

    for i, et in enumerate(results, 1):
        s = et.scores
        path_str = " → ".join(et.trajectory.nodes)
        pareto_badge = "⭐" * max(0, 3 - et.pareto_rank) or "·"

        table.add_row(
            str(i),
            path_str,
            f"${s.get('final_salary', 0):,.0f}",
            f"{s.get('salary_growth', 0):.1%}",
            f"{s.get('total_years', 0):.0f}",
            f"{s.get('avg_risk', 0):.0%}",
            f"{s.get('avg_satisfaction', 0):.0%}",
            pareto_badge,
        )

    console.print(table)


def print_profile_menu():
    """Muestra los perfiles disponibles y devuelve la clave seleccionada (o None)."""
    console.print(
        Panel(
            "\n".join(f"[bold]{k}[/bold] — {PROFILE_DESCRIPTIONS.get(k,'')}" for k in PROFILES),
            title="Perfiles disponibles",
            border_style="cyan",
        )
    )
    choices = list(PROFILES.keys()) + ["cancel"]
    choice = Prompt.ask("Seleccione perfil", choices=choices, default="balanced")
    if choice == "cancel":
        console.print("Operación cancelada.", style="dim")
        return None
    console.print(f"Perfil seleccionado: [bold]{choice}[/bold]")
    return choice
