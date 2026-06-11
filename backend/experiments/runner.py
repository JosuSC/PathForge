"""
experiments/runner.py
---------------------
Ejecuta el diseño experimental completo de PathForge.

Compara sistemáticamente:
    - 4 configuraciones del generador (beam_width, max_depth)
    - 4 perfiles de restricciones
    - 4 nodos de inicio (del grafo default o dinámicos del domain graph)
    
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# FIX [R1]: bootstrap de sys.path — mismo patrón que run_experiments.py
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

# FIX [R5]: importación condicional de rich — no rompe si no está instalado
try:
    from rich.console import Console
    from rich.progress import track
    _console = Console()
    _rich_available = True
except ImportError:
    _console = None
    _rich_available = False
    def track(iterable, **kwargs):  # type: ignore[misc]
        """Fallback sin rich: itera normalmente."""
        return iterable

# FIX [R1]: importaciones absolutas con prefijo backend
from backend.core.constraints import ConstraintProfiles, Constraint
from backend.core.generator import GeneratorConfig, TrajectoryGenerator
from backend.core.graph import CareerGraph
from backend.core.scorer import CareerOutcomePredictor        # FIX [R3]
from backend.data.loader import load_career_graph, load_domain_graph, list_available_domains
from backend.experiments.metrics import ExperimentMetrics, compute_metrics


# ---------------------------------------------------------------------------
# FIX [R4]: directorio de resultados unificado con run_experiments.py
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "results"

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

# Nodos de inicio para el grafo default (careers.json)
DEFAULT_SOURCE_NODES = [
    "junior_dev", "mid_dev", "data_scientist", "devops_engineer",
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _load_graph_and_sources(
    domain_id: str | None,
    predictor: CareerOutcomePredictor,
) -> tuple[CareerGraph, list[str]]:
    """
    Carga el grafo correcto y determina SOURCE_NODES dinámicamente.

    - Sin domain_id: grafo default (careers.json) + DEFAULT_SOURCE_NODES.
    - Con domain_id: domain graph + nodos de tipo 'entry' del grafo.
    """
    if domain_id:
        raw_graph = load_domain_graph(domain_id)
        graph     = CareerGraph(raw_graph, outcome_predictor=predictor)
        # Nodos 'entry' del domain graph como puntos de partida
        source_nodes = [
            nid for nid in graph.all_node_ids()
            if graph.node_attrs(nid).get("type") == "entry"
        ][:4]   # máximo 4 para mantener el mismo número que el default
        if not source_nodes:
            # Fallback: primeros 4 nodos si no hay 'entry' explícito
            source_nodes = graph.all_node_ids()[:4]
        logger.info(
            f"Domain graph '{domain_id}': {len(graph.all_node_ids())} nodos, "
            f"source_nodes={source_nodes}"
        )
    else:
        raw_graph    = load_career_graph()
        graph        = CareerGraph(raw_graph, outcome_predictor=predictor)  # FIX [R3]
        source_nodes = DEFAULT_SOURCE_NODES

    return graph, source_nodes


def run_all_experiments(
    domain_id: str | None = None,
) -> list[ExperimentMetrics]:
    """
    Ejecuta todas las combinaciones del experimento y guarda resultados.

    Args:
        domain_id: Si se pasa, usa ese domain graph en lugar de careers.json.
                   Permite experimentar con cualquier dominio real.

    Total de ejecuciones:
        4 configs × 4 perfiles × 4 fuentes = 64 experimentos
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # FIX [R3]: predictor instanciado como en main_api.py
    logger.info("Cargando modelo ML...")
    predictor = CareerOutcomePredictor.load_or_train()

    graph, source_nodes = _load_graph_and_sources(domain_id, predictor)

    all_metrics: list[ExperimentMetrics] = []
    combinations = [
        (cfg_name, cfg, prof_name, prof, source)
        for cfg_name, cfg in GENERATOR_CONFIGS.items()
        for prof_name, prof in CONSTRAINT_PROFILES.items()
        for source in source_nodes
    ]

    total = len(combinations)
    logger.info(f"Iniciando {total} experimentos...")

    for cfg_name, cfg, prof_name, prof, source in track(
        combinations, description="Ejecutando experimentos..."
    ):
        experiment_id = f"{cfg_name}__{prof_name}__{source}"

        try:
            # Verificar que el nodo existe antes de lanzar el generador
            if source not in graph.all_node_ids():
                logger.warning(
                    f"Nodo '{source}' no existe en el grafo. "
                    f"Saltando experimento {experiment_id}."
                )
                continue

            generator = TrajectoryGenerator(graph, cfg)

            start      = time.perf_counter()
            results    = generator.generate(source=source, constraints=prof)
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

    # Guardar resultados en JSON (FIX [R4]: mismo RESULTS_DIR)
    output_path = RESULTS_DIR / "experiment_results.json"
    data        = [m.to_dict() for m in all_metrics]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.success(
        f"Resultados guardados en {output_path} "
        f"({len(all_metrics)}/{total} experimentos completados)"
    )

    _print_summary(all_metrics)
    return all_metrics


def _print_summary(metrics: list[ExperimentMetrics]) -> None:
    """
    Imprime un resumen ejecutivo. FIX [R6]: protegido si metrics está vacío.
    """
    if not metrics:
        logger.warning("No hay métricas para resumir.")
        return

    def _rich_print(msg: str) -> None:
        if _rich_available and _console:
            _console.print(msg)
        else:
            logger.info(msg.replace("[bold cyan]","").replace("[/bold cyan]","")
                          .replace("[green]","").replace("[/green]","")
                          .replace("[yellow]","").replace("[/yellow]",""))

    _rich_print("\n[bold cyan]═══ RESUMEN EXPERIMENTAL ═══[/bold cyan]")
    _rich_print(f"  Experimentos completados: [green]{len(metrics)}[/green]")

    # FIX [R6]: max/min protegidos — metrics ya verificado no vacío arriba
    best_diversity = max(metrics, key=lambda m: m.diversity_score)
    best_pareto    = max(metrics, key=lambda m: m.pareto_front_size)
    fastest        = min(metrics, key=lambda m: m.execution_time_ms)
    best_terminal  = max(metrics, key=lambda m: m.terminal_rate)

    _rich_print(
        f"  Mayor diversidad    : [yellow]{best_diversity.config_name}[/yellow] "
        f"+ [yellow]{best_diversity.constraint_profile}[/yellow] "
        f"(score={best_diversity.diversity_score:.3f})"
    )
    _rich_print(
        f"  Mayor frente Pareto : [yellow]{best_pareto.config_name}[/yellow] "
        f"+ [yellow]{best_pareto.constraint_profile}[/yellow] "
        f"(size={best_pareto.pareto_front_size})"
    )
    _rich_print(
        f"  Mayor tasa terminal : [yellow]{best_terminal.config_name}[/yellow] "
        f"+ [yellow]{best_terminal.constraint_profile}[/yellow] "
        f"({best_terminal.terminal_rate:.1%})"
    )
    _rich_print(
        f"  Más rápido          : [yellow]{fastest.config_name}[/yellow] "
        f"({fastest.execution_time_ms:.1f}ms)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PathForge — Diseño Experimental")
    parser.add_argument(
        "--domain", type=str, default=None,
        help="ID del domain graph a usar (ej: software_development). "
             "Por defecto usa careers.json."
    )
    args = parser.parse_args()
    run_all_experiments(domain_id=args.domain)
