"""
experiments/visualizer.py
-------------------------
Genera gráficas del diseño experimental para el informe técnico.

Produce 4 gráficas:
    1. Diversidad por configuración del generador
    2. Tamaño del frente de Pareto por perfil de restricciones
    3. Trade-off crecimiento salarial vs riesgo
    4. Tiempo de ejecución por configuración
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_PATH = Path(__file__).parent / "results" / "experiment_results.json"
PLOTS_DIR    = Path(__file__).parent / "results" / "plots"


def load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontraron resultados. Corre primero experiments/runner.py"
        )
    return json.loads(RESULTS_PATH.read_text())


def plot_all() -> None:
    """Genera y guarda todas las gráficas del experimento."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PathForge — Resultados Experimentales", fontsize=16, fontweight="bold")

    _plot_diversity_by_config(axes[0, 0], data)
    _plot_pareto_by_profile(axes[0, 1], data)
    _plot_salary_vs_risk(axes[1, 0], data)
    _plot_execution_time(axes[1, 1], data)

    plt.tight_layout()
    output = PLOTS_DIR / "experiment_summary.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráficas guardadas en {output}")


# ---------------------------------------------------------------------------
# Gráficas individuales
# ---------------------------------------------------------------------------

def _plot_diversity_by_config(ax: plt.Axes, data: list[dict]) -> None:
    configs = ["narrow_shallow", "narrow_deep", "wide_shallow", "wide_deep"]
    means = [
        np.mean([d["diversity"] for d in data if d["config"] == c]) or 0
        for c in configs
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(configs, means, color=colors, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title("Diversidad promedio por configuración", fontweight="bold")
    ax.set_ylabel("Diversity Score")
    ax.set_ylim(0, max(means) * 1.2 if means else 1)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)


def _plot_pareto_by_profile(ax: plt.Axes, data: list[dict]) -> None:
    profiles = ["conservative", "ambitious", "balanced", "fast_track"]
    means = [
        np.mean([d["pareto_front_size"] for d in data if d["profile"] == p]) or 0
        for p in profiles
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(profiles, means, color=colors, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_title("Tamaño del frente de Pareto por perfil", fontweight="bold")
    ax.set_ylabel("Trayectorias en frente Pareto (promedio)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)


def _plot_salary_vs_risk(ax: plt.Axes, data: list[dict]) -> None:
    profiles = ["conservative", "ambitious", "balanced", "fast_track"]
    colors   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for profile, color in zip(profiles, colors):
        subset = [d for d in data if d["profile"] == profile]
        if not subset:
            continue
        x = [d["avg_risk"] for d in subset]
        y = [d["avg_salary_growth"] for d in subset]
        ax.scatter(x, y, c=color, alpha=0.6, s=40, label=profile)

    ax.set_title("Trade-off: Crecimiento salarial vs Riesgo", fontweight="bold")
    ax.set_xlabel("Riesgo promedio")
    ax.set_ylabel("Crecimiento salarial promedio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _plot_execution_time(ax: plt.Axes, data: list[dict]) -> None:
    configs = ["narrow_shallow", "narrow_deep", "wide_shallow", "wide_deep"]
    means = [
        np.mean([d["time_ms"] for d in data if d["config"] == c]) or 0
        for c in configs
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(configs, means, color=colors, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%.1fms", padding=3, fontsize=9)
    ax.set_title("Tiempo de ejecución por configuración", fontweight="bold")
    ax.set_ylabel("Tiempo (ms)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)


if __name__ == "__main__":
    plot_all()
