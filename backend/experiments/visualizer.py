"""
experiments/visualizer.py
-------------------------
Genera gráficas del diseño experimental para el informe técnico.

Produce 5 gráficas:
    1. Diversidad por configuración del generador
    2. Tamaño del frente de Pareto por perfil de restricciones
    3. Trade-off crecimiento salarial vs riesgo
    4. Tiempo de ejecución por configuración
    5. [NUEVA] Tasa de terminales alcanzados por configuración
    
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # backend sin pantalla — funciona en servidor/CI
import matplotlib.pyplot as plt
import numpy as np

# FIX [V4]: constantes centralizadas
PROJECT_NAME = "PathForge"
PALETTE      = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

RESULTS_DIR  = Path(__file__).resolve().parent / "results"
PLOTS_DIR    = RESULTS_DIR / "plots"

# FIX [R4]: rutas de ambas fuentes unificadas
_RUNNER_JSON      = RESULTS_DIR / "experiment_results.json"
_RUN_EXP_JSON     = RESULTS_DIR / "run_experiments_results.json"


# ---------------------------------------------------------------------------
# Carga de resultados
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    """FIX [V1]: mean seguro — retorna 0.0 si la lista está vacía."""
    return float(np.mean(values)) if values else 0.0


def _safe_ylim(ax: plt.Axes, means: list[float], margin: float = 1.2) -> None:
    """FIX [V2]: ylim robusto en todas las gráficas."""
    max_val = max(means) if means else 0.0
    ax.set_ylim(0, max_val * margin if max_val > 0 else 1.0)


def load_results() -> list[dict[str, Any]]:
    """
    Carga y unifica resultados de ambas fuentes:
        - experiment_results.json    (runner.py)
        - run_experiments_results.json (run_experiments.py)

    Normaliza los campos para que ambos formatos sean comparables.
    """
    data: list[dict] = []

    for path in (_RUNNER_JSON, _RUN_EXP_JSON):
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                # Normalizar: run_experiments usa 'beam_width'+'max_depth' en lugar de 'config'
                for entry in raw:
                    if "config" not in entry and "beam_width" in entry:
                        entry["config"] = f"beam{entry['beam_width']}_depth{entry['max_depth']}"
                    if "source" not in entry and "source_career" in entry:
                        entry["source"] = entry["source_career"]
                data.extend(raw)
            except Exception as e:
                print(f"[WARN] No se pudo cargar {path.name}: {e}")

    if not data:
        raise FileNotFoundError(
            f"No se encontraron resultados en {RESULTS_DIR}.\n"
            f"Ejecuta primero:\n"
            f"  python experiments/runner.py\n"
            f"  python experiments/run_experiments.py"
        )

    print(f"[INFO] {len(data)} registros cargados desde {RESULTS_DIR}")
    return data


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def plot_all(data: list[dict] | None = None) -> None:
    """Genera y guarda todas las gráficas del experimento."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = load_results()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{PROJECT_NAME} — Resultados Experimentales",   # FIX [V4]
        fontsize=16, fontweight="bold"
    )

    _plot_diversity_by_config(axes[0, 0], data)
    _plot_pareto_by_profile(axes[0, 1], data)
    _plot_salary_vs_risk(axes[0, 2], data)
    _plot_execution_time(axes[1, 0], data)
    _plot_terminal_rate(axes[1, 1], data)        # NUEVA gráfica
    _plot_transition_prob(axes[1, 2], data)      # NUEVA gráfica

    plt.tight_layout()
    output = PLOTS_DIR / "experiment_summary.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gráficas guardadas en {output}")


def _bar_chart(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    fmt: str = "%.3f",
) -> None:
    """Helper: gráfica de barras con FIX [V1] y FIX [V2] aplicados."""
    # FIX [V1]: valores ya son seguros (calculados con _safe_mean)
    bars = ax.bar(labels, values, color=PALETTE[:len(labels)],  # FIX [V4]
                  alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    _safe_ylim(ax, values)                       # FIX [V2]
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)


def _plot_diversity_by_config(ax: plt.Axes, data: list[dict]) -> None:
    configs = sorted({d["config"] for d in data if "config" in d})
    # FIX [V1]: _safe_mean nunca crashea con lista vacía
    means   = [_safe_mean([d["diversity"] for d in data if d.get("config") == c])
               for c in configs]
    _bar_chart(ax, configs, means,
               "Diversidad promedio por configuración", "Diversity Score")


def _plot_pareto_by_profile(ax: plt.Axes, data: list[dict]) -> None:
    profiles = ["conservative", "ambitious", "balanced", "fast_track"]
    means    = [_safe_mean([d["pareto_front_size"] for d in data
                            if d.get("profile") == p])
                for p in profiles]
    _bar_chart(ax, profiles, means,
               "Tamaño del frente de Pareto por perfil",
               "Trayectorias en frente Pareto (promedio)", fmt="%.1f")


def _plot_salary_vs_risk(ax: plt.Axes, data: list[dict]) -> None:
    profiles = ["conservative", "ambitious", "balanced", "fast_track"]
    for profile, color in zip(profiles, PALETTE):  # FIX [V4]
        subset = [d for d in data if d.get("profile") == profile]
        if not subset:
            continue
        x = [d.get("avg_risk",          d.get("avg_risk", 0)) for d in subset]
        y = [d.get("avg_salary_growth",  0) for d in subset]
        # FIX [V1]: no crashea si subset está vacío (guardado arriba)
        ax.scatter(x, y, c=color, alpha=0.6, s=40, label=profile)

    ax.set_title("Trade-off: Crecimiento salarial vs Riesgo", fontweight="bold")
    ax.set_xlabel("Riesgo promedio")
    ax.set_ylabel("Crecimiento salarial promedio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _plot_execution_time(ax: plt.Axes, data: list[dict]) -> None:
    configs = sorted({d["config"] for d in data if "config" in d})
    means   = [_safe_mean([d["time_ms"] for d in data if d.get("config") == c])
               for c in configs]
    _bar_chart(ax, configs, means,
               "Tiempo de ejecución por configuración", "Tiempo (ms)", fmt="%.1fms")


def _plot_terminal_rate(ax: plt.Axes, data: list[dict]) -> None:
    """
    Tasa de trayectorias que alcanzan un nodo terminal,
    usando el campo terminal_rate de ExperimentMetrics v2 (graph.py fix G1).
    """
    configs = sorted({d["config"] for d in data if "config" in d})
    means   = [_safe_mean([d.get("terminal_rate", 0.0) for d in data
                           if d.get("config") == c])
               for c in configs]
    _bar_chart(ax, configs, means,
               "Tasa de terminales alcanzados", "Terminal Rate", fmt="%.2f")


def _plot_transition_prob(ax: plt.Axes, data: list[dict]) -> None:
    """
    Probabilidad media de transición por perfil.
    Usa avg_transition_prob de ExperimentMetrics v2 (graph.py fix G3).
    """
    profiles = ["conservative", "ambitious", "balanced", "fast_track"]
    means    = [_safe_mean([d.get("avg_transition_prob", 0.0) for d in data
                            if d.get("profile") == p])
                for p in profiles]
    _bar_chart(ax, profiles, means,
               "Probabilidad media de transición por perfil",
               "Avg Transition Probability", fmt="%.3f")


if __name__ == "__main__":
    plot_all()
