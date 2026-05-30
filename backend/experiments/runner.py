"""
experiments/runner.py
---------------------
Ejecuta el diseño experimental completo de PathForge.

Compara sistemáticamente:
    - 4 configuraciones del generador (beam_width, max_depth)
    - 4 perfiles de restricciones
    - 4 nodos de inicio distintos

Genera resultados en JSON para posterior análisis y visualización.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import track

from core.constraints import ConstraintProfiles, Constraint
from core.generator import GeneratorConfig, TrajectoryGenerator
from core.graph import CareerGraph
from data.loader import load_career_graph
from experiments.metrics import ExperimentMetrics, compute_metrics

console = Console()

# ---------------------------------------------------------------------------
# Configuraciones a comparar
# ---------------------------------------------------------------------------

GENERATOR_CONFIGS: dict[str, GeneratorConfig] = {
    "narrow_shallow": GeneratorConfig(beam_width=4,  max_depth=3, top_k_results=10),
    "narrow_deep":    GeneratorConfig(beam_width=4,  max_depth=6, top_k_results=10),
    "wide_shallow":   GeneratorConfig(beam_width=12, max_depth=3, top_k_results=10),
    "wide_deep":      GeneratorConfig(beam_width=12, max_depth=6, top_k_results=10),
}

CONSTRAINT_PROFILES: dict[str, Constraint] = {
    "conservative": ConstraintProfiles.conservative(),
    "ambitious":    ConstraintProfiles.ambitious(),
    "balanced":     ConstraintProfiles.balanced(),
    "fast_track":   ConstraintProfiles.fast_track(),
}

SOURCE_NODES = ["junior_dev", "mid_dev", "data_scientist", "devops_engineer"]

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_experiments() -> list[ExperimentMetrics]:
    """
    Ejecuta todas las combinaciones del experimento y guarda resultados.

    Total de ejecuciones:
        4 configs × 4 perfiles × 4 fuentes = 64 experimentos
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    G = load_career_graph()
    graph = CareerGraph(G)

    all_metrics: list[ExperimentMetrics] = []

    total = len(GENERATOR_CONFIGS) * len(CONSTRAINT_PROFILES) * len(SOURCE_NODES)
    logger.info(f"Iniciando {total} experimentos...")

    combinations = [
        (cfg_name, cfg, prof_name, prof, source)
        for cfg_name, cfg in GENERATOR_CONFIGS.items()
        for prof_name, prof in CONSTRAINT_PROFILES.items()
        for source in SOURCE_NODES
    ]

    for cfg_name, cfg, prof_name, prof, source in track(
        combinations, description="Ejecutando experimentos..."
    ):
        experiment_id = f"{cfg_name}__{prof_name}__{source}"

        try:
            generator = TrajectoryGenerator(graph, cfg)

            start = time.perf_counter()
            results = generator.generate(source=source, constraints=prof)
            elapsed_ms = (time.perf_counter() - start) * 1000

            metrics = compute_metrics(
                results=results,
                config_name=cfg_name,
                source_node=source,
                constraint_profile=prof_name,
                execution_time_ms=elapsed_ms,
            )
            all_metrics.append(metrics)

        except Exception as exc:
            logger.warning(f"Experimento {experiment_id} falló: {exc}")

    # Guardar resultados
    output_path = RESULTS_DIR / "experiment_results.json"
    data = [m.to_dict() for m in all_metrics]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.success(f"Resultados guardados en {output_path}")

    _print_summary(all_metrics)
    return all_metrics


def _print_summary(metrics: list[ExperimentMetrics]) -> None:
    """Imprime un resumen ejecutivo de los experimentos."""
    if not metrics:
        return

    console.print("\n[bold cyan]═══ RESUMEN EXPERIMENTAL ═══[/bold cyan]")
    console.print(f"  Experimentos ejecutados : [green]{len(metrics)}[/green]")

    best_diversity = max(metrics, key=lambda m: m.diversity_score)
    best_pareto    = max(metrics, key=lambda m: m.pareto_front_size)
    fastest        = min(metrics, key=lambda m: m.execution_time_ms)

    console.print(
        f"  Mayor diversidad        : [yellow]{best_diversity.config_name}[/yellow] "
        f"+ [yellow]{best_diversity.constraint_profile}[/yellow] "
        f"(score={best_diversity.diversity_score:.3f})"
    )
    console.print(
        f"  Mayor frente Pareto     : [yellow]{best_pareto.config_name}[/yellow] "
        f"+ [yellow]{best_pareto.constraint_profile}[/yellow] "
        f"(size={best_pareto.pareto_front_size})"
    )
    console.print(
        f"  Más rápido              : [yellow]{fastest.config_name}[/yellow] "
        f"({fastest.execution_time_ms:.1f}ms)"
    )


if __name__ == "__main__":
    run_all_experiments()
