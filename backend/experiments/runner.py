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
import argparse

# Bootstrap de sys.path — mismo patrón que run_experiments.py
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

# Importación condicional de rich — no rompe si no está instalado
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

# Importaciones absolutas con prefijo backend
from backend.core.constraints import ConstraintProfiles, Constraint
from backend.core.generator import GeneratorConfig, TrajectoryGenerator
from backend.core.graph import CareerGraph
from backend.core.scorer import CareerOutcomePredictor        # FIX [R3]
from backend.data.loader import load_career_graph, load_domain_graph, list_available_domains
from backend.experiments.metrics import ExperimentMetrics, compute_metrics


# ---------------------------------------------------------------------------
# Directorio de resultados unificado con run_experiments.py
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
    instances_path: Path | None = None,
) -> tuple[CareerGraph, list[dict]]:
    """
    Carga el grafo y retorna:
    - Si instances_path existe: (graph, lista de instancias desde el archivo)
    - Si no: (graph, lista de instancias auto-generadas)
    """
    # 1. Cargar el grafo
    if domain_id:
        raw_graph = load_domain_graph(domain_id)
        graph = CareerGraph(raw_graph, outcome_predictor=predictor)
    else:
        raw_graph = load_career_graph()
        graph = CareerGraph(raw_graph, outcome_predictor=predictor)

    # 2. Si hay archivo de instancias, usarlo
    if instances_path is not None and instances_path.exists():
        logger.info(f"Cargando instancias desde {instances_path}")
        instances = json.loads(instances_path.read_text(encoding="utf-8"))
        return graph, instances

    # 3. No hay archivo -> generar instancias automáticas
    if domain_id:
        # Estrategia inteligente: priorizar nodos con sucesores que no sean terminales
        all_ids = graph.all_node_ids()
        non_terminal = [nid for nid in all_ids if not graph.is_terminal(nid)]

        # Prioridad 1: entry con al menos 2 sucesores
        entry_with_succ = [
            nid for nid in non_terminal
            if graph.node_attrs(nid).get("type") == "entry"
            and len(graph.successors(nid)) >= 2
        ]
        # Prioridad 2: mid con al menos 2 sucesores
        mid_with_succ = [
            nid for nid in non_terminal
            if graph.node_attrs(nid).get("type") == "mid"
            and len(graph.successors(nid)) >= 2
        ]
        # Prioridad 3: cualquier nodo no-terminal con al menos 2 sucesores
        any_with_succ = [
            nid for nid in non_terminal
            if len(graph.successors(nid)) >= 2
        ]
        # Prioridad 4: cualquier nodo no-terminal con al menos 1 sucesor
        any_one_succ = [
            nid for nid in non_terminal
            if len(graph.successors(nid)) >= 1
        ]

        source_nodes = (entry_with_succ + mid_with_succ + any_with_succ + any_one_succ)[:4]

        # Fallback último: los primeros 4 IDs
        if not source_nodes:
            source_nodes = all_ids[:4]

        logger.info(
            f"Domain graph '{domain_id}': source_nodes={source_nodes} "
            f"(entry={len(entry_with_succ)}, mid={len(mid_with_succ)}, "
            f"any2={len(any_with_succ)}, any1={len(any_one_succ)})"
        )
    else:
        source_nodes = DEFAULT_SOURCE_NODES

    auto_instances = []
    for source in source_nodes:
        for profile_name in CONSTRAINT_PROFILES:
            auto_instances.append({
                "source_career": source,
                "profile": profile_name,
            })
    return graph, auto_instances


def run_all_experiments(
    domain_id: str | None = None,
    instances_path: Path | None = None,
) -> list[ExperimentMetrics]:
    """
    Ejecuta todas las combinaciones del experimento y guarda resultados.

    Si se proporciona instances_path, carga las instancias desde ese archivo.
    Cada instancia es un dict con:
        - "source_career": str
        - "profile": str (nombre del perfil, ej. "balanced")
        - "beam_width": int (opcional)
        - "max_depth": int (opcional)

    Args:
        domain_id: Si se pasa, usa ese domain graph en lugar de careers.json.
        instances_path: Ruta al archivo JSON con instancias.

    Returns:
        Lista de métricas de cada experimento completado.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando modelo ML...")
    predictor = CareerOutcomePredictor.load_or_train()

    # Carga el grafo y las instancias (auto-generadas o desde archivo)
    graph, instances = _load_graph_and_sources(domain_id, predictor, instances_path)

    all_metrics: list[ExperimentMetrics] = []
    combinations = []

    # Perfil por defecto por si el nombre no existe
    default_profile = ConstraintProfiles.balanced()

    for inst in instances:
        source = inst["source_career"]
        profile_name = inst.get("profile", "balanced")
        # Obtener el objeto ConstraintProfile
        prof = CONSTRAINT_PROFILES.get(profile_name, default_profile)

        # Caso 1: instancia define sus propios parámetros de generación
        bw = inst.get("beam_width")
        md = inst.get("max_depth")
        if bw is not None and md is not None:
            cfg = GeneratorConfig(beam_width=bw, max_depth=md, top_k_results=10)
            cfg_name = f"bw{bw}_md{md}"
            combinations.append((cfg_name, cfg, profile_name, prof, source))
        else:
            # Caso 2: usar todas las configuraciones globales
            for cfg_name, cfg in GENERATOR_CONFIGS.items():
                combinations.append((cfg_name, cfg, profile_name, prof, source))

    total = len(combinations)
    logger.info(f"Iniciando {total} experimentos...")

    for cfg_name, cfg, prof_name, prof, source in track(
        combinations, description="Ejecutando experimentos..."
    ):
        experiment_id = f"{cfg_name}__{prof_name}__{source}"

        try:
            if source not in graph.all_node_ids():
                logger.warning(
                    f"Nodo '{source}' no existe en el grafo. "
                    f"Saltando experimento {experiment_id}."
                )
                continue

            generator = TrajectoryGenerator(graph, cfg)

            start      = time.perf_counter()
            results    = generator.generate(source=source, constraints=prof)
            
            # Si no hay trayectorias, saltar el cálculo de métricas
            if not results:
                logger.warning(f"Sin trayectorias generadas para {experiment_id}. Saltando métricas.")
                continue
                
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

    # Guardar resultados en JSON e Incluir el nombre del dominio en el archivo
    filename = f"experiment_results_{domain_id}.json" if domain_id else "experiment_results.json"
    output_path = RESULTS_DIR / filename
    data = [m.to_dict() for m in all_metrics]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
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

    # max/min protegidos — metrics ya verificado no vacío arriba
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
    parser = argparse.ArgumentParser(description="PathForge — Diseño Experimental")
    
    parser.add_argument(
        "--domain", type=str, default=None,
        help="ID del domain graph a usar (ej: software_development). "
             "Por defecto usa careers.json."
    )
    parser.add_argument(
        "--instances", type=str, default=None,
        help="Ruta al archivo JSON con instancias personalizadas. "
             "Si no se pasa, se usan las instancias por defecto."
    )
    
    args = parser.parse_args()

    # Conversión segura a Path
    inst_path = Path(args.instances) if args.instances else None
    
    run_all_experiments(domain_id=args.domain, instances_path=inst_path)
