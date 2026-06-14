"""
experiments/visualizer.py
-------------------------
Genera graficas del diseno experimental para el informe tecnico.

Lee datos desde backend/experiments/results/ (CSV y JSON generados
por run_experiments.py). Produce graficas publication-ready de
TODOS los resultados para el informe.

Graficas generadas:
    1. Tasa de exito por dominio
    2. Diversidad por configuracion del generador
    3. Success Score (Monte Carlo) por perfil de restricciones
    4. Trade-off crecimiento salarial vs riesgo (scatter)
    5. Tiempo de ejecucion por configuracion
    6. Tasa de terminales alcanzados por perfil
    7. Simulacion MC: distribucion de success_score por perfil (boxplot)
    8. Comparacion salary_growth vs sim_salary_p50
    9. Eventos de riesgo por perfil (desde simulacion)
    10. Heatmap dominio x perfil (success rate)

Uso:
    python -m backend.experiments.visualizer
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Cargar fuentes
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_NAME = "PathForge"
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
           "#64B5CD", "#8C8C8C", "#E5AE38", "#6D904F", "#D65F5F"]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"

CSV_PATH      = RESULTS_DIR / "full_experiment_results.csv"
JSON_PATH     = RESULTS_DIR / "full_experiment_results.json"
SUMMARY_PATH  = RESULTS_DIR / "experiment_summary.json"


# ---------------------------------------------------------------------------
# Carga de resultados
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        val = float(v)
        return val if np.isfinite(val) else default
    except (ValueError, TypeError):
        return default


def load_results_from_csv() -> list[dict[str, Any]]:
    """Carga resultados desde el CSV generado por run_experiments.py."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {CSV_PATH}.\n"
            f"Ejecuta primero: python -m backend.experiments.run_experiments"
        )

    data: list[dict] = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed = {}
            for k, v in row.items():
                typed[k] = v
            data.append(typed)

    print(f"[INFO] {len(data)} registros cargados desde {CSV_PATH.name}")
    return data


def load_results_from_json() -> list[dict[str, Any]]:
    """Carga resultados desde el JSON generado por run_experiments.py (fallback)."""
    if not JSON_PATH.exists():
        return []
    try:
        raw = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        print(f"[INFO] {len(raw)} registros cargados desde {JSON_PATH.name}")
        return raw
    except Exception as e:
        print(f"[WARN] No se pudo cargar {JSON_PATH.name}: {e}")
        return []


def load_all_results() -> list[dict[str, Any]]:
    """Carga resultados desde CSV primero, JSON como fallback."""
    data = load_results_from_csv()
    if not data:
        data = load_results_from_json()
    if not data:
        # Intentar buscar otros JSONs en results/
        for json_file in RESULTS_DIR.glob("*.json"):
            if json_file.name == "experiment_summary.json":
                continue
            try:
                raw = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(raw, list) and raw:
                    print(f"[INFO] {len(raw)} registros cargados desde {json_file.name}")
                    data.extend(raw)
            except Exception:
                pass
    if not data:
        raise FileNotFoundError(
            f"No se encontraron resultados en {RESULTS_DIR}.\n"
            f"Ejecuta primero: python -m backend.experiments.run_experiments"
        )
    return data


def load_summary() -> dict[str, Any]:
    """Carga el resumen por dominio."""
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_ylim(ax: plt.Axes, means: list[float], margin: float = 1.2) -> None:
    max_val = max(means) if means else 0.0
    ax.set_ylim(0, max_val * margin if max_val > 0 else 1.0)


def _bar_chart(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    fmt: str = "%.3f",
    colors: list[str] | None = None,
) -> None:
    palette = colors or PALETTE[:len(labels)]
    bars = ax.bar(labels, values, color=palette, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt=fmt, padding=3, fontsize=8)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    _safe_ylim(ax, values)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------

def plot_all() -> None:
    """Genera y guarda todas las graficas."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all_results()

    # Parsear campos numericos desde CSV (strings -> floats)
    for row in data:
        for key in ["feasible"]:
            row[key] = row.get(key, "False") == "True"
        for key in ["computation_time_ms", "num_paths_found", "pareto_front_size",
                     "best_salary_growth", "best_final_salary", "best_satisfaction",
                     "avg_risk", "avg_years", "terminal_rate", "diversity_score",
                     "sim_success_score_mean", "sim_success_score_std",
                     "sim_salary_p50", "sim_salary_p90", "sim_years_mean",
                     "sim_risk_event_rate", "best_transition_prob", "best_avg_demand",
                     "pareto_rank_of_best", "beam_width", "max_depth"]:
            row[key] = _safe_float(row.get(key, 0))

    feasible = [r for r in data if r["feasible"]]
    print(f"[INFO] {len(feasible)} factibles de {len(data)} total")

    # --- Figura 1: Overview (6 graficas) ---
    fig1, axes1 = plt.subplots(2, 3, figsize=(20, 11))
    fig1.suptitle(f"{PROJECT_NAME} -- Resultados Experimentales (Overview)",
                  fontsize=16, fontweight="bold")

    _plot_success_rate_by_domain(axes1[0, 0], data)
    _plot_diversity_by_config(axes1[0, 1], feasible)
    _plot_sim_score_by_profile(axes1[0, 2], feasible)
    _plot_salary_vs_risk(axes1[1, 0], feasible)
    _plot_execution_time(axes1[1, 1], feasible)
    _plot_terminal_rate(axes1[1, 2], feasible)

    plt.tight_layout()
    fig1.savefig(PLOTS_DIR / "01_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[OK] 01_overview.png")

    # --- Figura 2: Simulacion Monte Carlo (4 graficas) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
    fig2.suptitle(f"{PROJECT_NAME} -- Simulacion Monte Carlo",
                  fontsize=16, fontweight="bold")

    _plot_sim_score_distribution(axes2[0, 0], feasible)
    _plot_salary_vs_sim_salary(axes2[0, 1], feasible)
    _plot_risk_events_by_profile(axes2[1, 0], feasible)
    _plot_sim_years_by_config(axes2[1, 1], feasible)

    plt.tight_layout()
    fig2.savefig(PLOTS_DIR / "02_monte_carlo.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK] 02_monte_carlo.png")

    # --- Figura 3: Heatmap y comparaciones (2 graficas) ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(18, 8))
    fig3.suptitle(f"{PROJECT_NAME} -- Comparaciones Cruzadas",
                  fontsize=16, fontweight="bold")

    _plot_heatmap_success_rate(axes3[0], data)
    _plot_beam_depth_effect(axes3[1], feasible)

    plt.tight_layout()
    fig3.savefig(PLOTS_DIR / "03_cross_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"[OK] 03_cross_comparison.png")

    print(f"\n[OK] Todas las graficas guardadas en {PLOTS_DIR}")


# ---------------------------------------------------------------------------
# Graficas individuales
# ---------------------------------------------------------------------------

def _plot_success_rate_by_domain(ax: plt.Axes, data: list[dict]) -> None:
    """Tasa de exito (factibles/total) por dominio."""
    domains = sorted(set(r["domain"] for r in data))
    rates = []
    labels = []
    for d in domains:
        d_rows = [r for r in data if r["domain"] == d]
        n_feas = len([r for r in d_rows if r["feasible"]])
        rate = n_feas / len(d_rows) if d_rows else 0
        rates.append(rate)
        labels.append(d[:15])

    colors = ["#55A868" if r >= 0.7 else "#E5AE38" if r >= 0.4 else "#C44E52" for r in rates]
    bars = ax.bar(labels, rates, color=colors, alpha=0.85, edgecolor="white")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate*100:.0f}%", ha='center', va='bottom', fontsize=7)
    ax.set_title("Tasa de Exito por Dominio", fontweight="bold", fontsize=10)
    ax.set_ylabel("Tasa de Exito", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.7, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.3)


def _plot_diversity_by_config(ax: plt.Axes, data: list[dict]) -> None:
    configs = sorted(set(r.get("config_name", "") for r in data if r.get("config_name")))
    means = [_safe_mean([r["diversity_score"] for r in data if r.get("config_name") == c])
             for c in configs]
    _bar_chart(ax, configs, means,
               "Diversidad promedio por configuracion", "Diversity Score")


def _plot_sim_score_by_profile(ax: plt.Axes, data: list[dict]) -> None:
    profiles = ["conservative", "ambitious", "balanced", "fast_track"]
    means = [_safe_mean([r["sim_success_score_mean"] for r in data
                         if r.get("profile") == p and r["sim_success_score_mean"] > 0])
             for p in profiles]
    _bar_chart(ax, profiles, means,
               "Success Score (Monte Carlo) por perfil", "Avg Sim Score", fmt="%.3f")


def _plot_salary_vs_risk(ax: plt.Axes, data: list[dict]) -> None:
    profiles = ["conservative", "ambitious", "balanced", "fast_track"]
    for profile, color in zip(profiles, PALETTE[:4]):
        subset = [r for r in data if r.get("profile") == profile]
        if not subset:
            continue
        x = [r["avg_risk"] for r in subset]
        y = [r["best_salary_growth"] for r in subset]
        ax.scatter(x, y, c=color, alpha=0.5, s=30, label=profile)

    ax.set_title("Trade-off: Crecimiento Salarial vs Riesgo", fontweight="bold", fontsize=10)
    ax.set_xlabel("Riesgo promedio", fontsize=9)
    ax.set_ylabel("Crecimiento salarial", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)


def _plot_execution_time(ax: plt.Axes, data: list[dict]) -> None:
    configs = sorted(set(r.get("config_name", "") for r in data if r.get("config_name")))
    means = [_safe_mean([r["computation_time_ms"] for r in data if r.get("config_name") == c])
             for c in configs]
    _bar_chart(ax, configs, means,
               "Tiempo de ejecucion por configuracion", "Tiempo (ms)", fmt="%.1f")


def _plot_terminal_rate(ax: plt.Axes, data: list[dict]) -> None:
    profiles = ["conservative", "ambitious", "balanced", "fast_track"]
    means = [_safe_mean([r["terminal_rate"] for r in data if r.get("profile") == p])
             for p in profiles]
    _bar_chart(ax, profiles, means,
               "Tasa de terminales alcanzados por perfil", "Terminal Rate", fmt="%.2f")


def _plot_sim_score_distribution(ax: plt.Axes, data: list[dict]) -> None:
    """Box plot de success_score (MC) por perfil."""
    profiles = ["conservative", "ambitious", "balanced", "fast_track"]
    box_data = []
    labels = []
    for p in profiles:
        vals = [r["sim_success_score_mean"] for r in data
                if r.get("profile") == p and r["sim_success_score_mean"] > 0]
        if vals:
            box_data.append(vals)
            labels.append(p)

    if box_data:
        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], PALETTE[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_title("Distribucion de Sim Score por perfil", fontweight="bold", fontsize=10)
    ax.set_ylabel("Success Score (Monte Carlo)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def _plot_salary_vs_sim_salary(ax: plt.Axes, data: list[dict]) -> None:
    """Scatter: best_final_salary vs sim_salary_p50."""
    x = [r["best_final_salary"] for r in data if r["sim_salary_p50"] > 0]
    y = [r["sim_salary_p50"] for r in data if r["sim_salary_p50"] > 0]
    if x:
        ax.scatter(x, y, alpha=0.5, s=30, c=PALETTE[0])
        max_val = max(max(x), max(y))
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=0.8)
    ax.set_title("Salario Final vs Salario Simulado (p50)", fontweight="bold", fontsize=10)
    ax.set_xlabel("Best Final Salary (determinista)", fontsize=9)
    ax.set_ylabel("Sim Salary p50 (Monte Carlo)", fontsize=9)
    ax.grid(alpha=0.3)


def _plot_risk_events_by_profile(ax: plt.Axes, data: list[dict]) -> None:
    """Tasa promedio de eventos de riesgo por perfil."""
    profiles = ["conservative", "ambitious", "balanced", "fast_track"]
    means = [_safe_mean([r["sim_risk_event_rate"] for r in data
                         if r.get("profile") == p and r["sim_risk_event_rate"] > 0])
             for p in profiles]
    _bar_chart(ax, profiles, means,
               "Eventos de Riesgo (MC) por perfil", "Avg Risk Event Rate", fmt="%.2f")


def _plot_sim_years_by_config(ax: plt.Axes, data: list[dict]) -> None:
    """Anos efectivos promedio (simulacion) por configuracion."""
    configs = sorted(set(r.get("config_name", "") for r in data if r.get("config_name")))
    means = [_safe_mean([r["sim_years_mean"] for r in data
                         if r.get("config_name") == c and r["sim_years_mean"] > 0])
             for c in configs]
    _bar_chart(ax, configs, means,
               "Anos efectivos (MC) por configuracion", "Avg Effective Years", fmt="%.1f")


def _plot_heatmap_success_rate(ax: plt.Axes, data: list[dict]) -> None:
    """Heatmap dominio x perfil con tasa de exito."""
    domains = sorted(set(r["domain"] for r in data))
    profiles = ["conservative", "ambitious", "balanced", "fast_track"]

    matrix = np.zeros((len(domains), len(profiles)))
    for i, d in enumerate(domains):
        for j, p in enumerate(profiles):
            subset = [r for r in data if r["domain"] == d and r.get("profile") == p]
            if subset:
                matrix[i, j] = len([r for r in subset if r["feasible"]]) / len(subset)

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(profiles)))
    ax.set_xticklabels(profiles, fontsize=8, rotation=20)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([d[:18] for d in domains], fontsize=7)

    for i in range(len(domains)):
        for j in range(len(profiles)):
            val = matrix[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_title("Tasa de Exito: Dominio x Perfil", fontweight="bold", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Tasa de exito")


def _plot_beam_depth_effect(ax: plt.Axes, data: list[dict]) -> None:
    """Efecto combinado de beam_width y max_depth en diversidad y tiempo."""
    bw_vals = sorted(set(r["beam_width"] for r in data if r["beam_width"] > 0))
    md_vals = sorted(set(r["max_depth"] for r in data if r["max_depth"] > 0))

    if not bw_vals or not md_vals:
        ax.text(0.5, 0.5, "Datos insuficientes", ha="center", transform=ax.transAxes)
        return

    for i, bw in enumerate(bw_vals):
        divs = []
        times = []
        labels = []
        for md in md_vals:
            subset = [r for r in data if r["beam_width"] == bw and r["max_depth"] == md and r["feasible"]]
            if subset:
                divs.append(_safe_mean([r["diversity_score"] for r in subset]))
                times.append(_safe_mean([r["computation_time_ms"] for r in subset]))
                labels.append(f"bw={int(bw)}\nd={int(md)}")

        if divs:
            color = PALETTE[i % len(PALETTE)]
            ax.scatter(times, divs, s=100, c=color, label=f"beam_width={int(bw)}",
                      alpha=0.8, edgecolors="white", linewidth=0.5)
            for t, d, l in zip(times, divs, labels):
                ax.annotate(l, (t, d), fontsize=7, ha="center", va="bottom")

    ax.set_title("Trade-off: Diversidad vs Tiempo", fontweight="bold", fontsize=10)
    ax.set_xlabel("Tiempo de ejecucion (ms)", fontsize=9)
    ax.set_ylabel("Diversidad", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    plot_all()
