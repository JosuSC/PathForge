"""
experiments/runner.py
---------------------
Ejecuta el diseno experimental completo de PathForge.

FIX ADAPTATIVO: Seleccion inteligente de source nodes que prioriza
nodos con >=2 sucesores. Ya no toma los primeros 4 "entry" (que en
domain graphs como engineering son 0). Estrategia: entry con sucesores
-> mid con sucesores -> cualquier nodo con sucesores.

FIX ENCODING: _print_summary usa solo ASCII para evitar
UnicodeEncodeError en Windows cp1252.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import argparse

# Bootstrap de sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

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

from backend.core.constraints import ConstraintProfiles, Constraint
from backend.core.generator import GeneratorConfig, TrajectoryGenerator
from backend.core.graph import CareerGraph
from backend.core.scorer import CareerOutcomePredictor
from backend.data.loader import load_career_graph, load_domain_graph, list_available_domains
from backend.experiments.metrics import ExperimentMetrics, compute_metrics


# ---------------------------------------------------------------------------
# Directorio de resultados
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
# FIX ADAPTATIVO: Seleccion inteligente de source nodes
# ---------------------------------------------------------------------------

def _select_source_nodes(graph: CareerGraph, max_nodes: int = 4) -> list[str]:
    """
    Selecciona nodos de inicio inteligentemente para domain graphs.

    Estrategia de prioridad:
    1. Nodos "entry" con >=2 sucesores
    2. Nodos "entry" con >=1 sucesor
    3. Nodos "mid" con >=2 sucesores
    4. Cualquier nodo con >=2 sucesores
    5. Los primeros nodos disponibles como ultimo recurso
    """
    all_nodes = graph.all_node_ids()

    def _filter_by_type_and_successors(
        node_type: str | None = None,
        min_successors: int = 2
    ) -> list[str]:
        result = []
        for nid in all_nodes:
            if node_type and graph.node_attrs(nid).get("type") != node_type:
                continue
            if len(graph.successors(nid)) >= min_successors:
                result.append(nid)
        return result

    # Estrategia 1: entry con >=2 sucesores
    candidates = _filter_by_type_and_successors("entry", 2)
    if len(candidates) >= max_nodes:
        return candidates[:max_nodes]

    # Estrategia 2: entry con >=1 sucesor
    candidates = _filter_by_type_and_successors("entry", 1)
    if len(candidates) >= max_nodes:
        return candidates[:max_nodes]

    # Estrategia 3: mid con >=2 sucesores
    mid_candidates = _filter_by_type_and_successors("mid", 2)
    candidates.extend(mid_candidates)
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    candidates = unique
    if len(candidates) >= max_nodes:
        return candidates[:max_nodes]

    # Estrategia 4: cualquier nodo con >=2 sucesores
    any_candidates = _filter_by_type_and_successors(None, 2)
    candidates.extend(any_candidates)
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    candidates = unique
    if len(candidates) >= max_nodes:
        return candidates[:max_nodes]

    # Estrategia 5: cualquier nodo con >=1 sucesor
    any1_candidates = _filter_by_type_and_successors(None, 1)
    candidates.extend(any1_candidates)
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    candidates = unique

    # Ultimo recurso: todos los nodos
    candidates.extend(all_nodes)
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    # Logging para debug
    entry_count = sum(1 for n in all_nodes if graph.node_attrs(n).get("type") == "entry")
    mid_with_2 = len(mid_candidates)
    any_with_2 = len(any_candidates)
    any_with_1 = len(any1_candidates)
    logger.info(
        f"Source selection: entry={entry_count}, mid(2+)={mid_with_2}, "
        f"any(2+)={any_with_2}, any(1+)={any_with_1}, "
        f"selected={unique[:max_nodes]}"
    )

    return unique[:max_nodes]


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
    if domain_id:
        raw_graph = load_domain_graph(domain_id)
        graph = CareerGraph(raw_graph, outcome_predictor=predictor)
    else:
        raw_graph = load_career_graph()
        graph = CareerGraph(raw_graph, outcome_predictor=predictor)

    if instances_path is not None and instances_path.exists():
        logger.info(f"Cargando instancias desde {instances_path}")
        instances = json.loads(instances_path.read_text(encoding="utf-8"))
        return graph, instances

    if domain_id:
        source_nodes = _select_source_nodes(graph, max_nodes=4)
        if not source_nodes:
            source_nodes = graph.all_node_ids()[:4]
        logger.info(f"Domain graph '{domain_id}': source_nodes={source_nodes}")
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
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando modelo ML...")
    predictor = CareerOutcomePredictor.load_or_train()

    graph, instances = _load_graph_and_sources(domain_id, predictor, instances_path)

    all_metrics: list[ExperimentMetrics] = []
    combinations = []

    default_profile = ConstraintProfiles.balanced()

    for inst in instances:
        source = inst["source_career"]
        profile_name = inst.get("profile", "balanced")
        prof = CONSTRAINT_PROFILES.get(profile_name, default_profile)

        bw = inst.get("beam_width")
        md = inst.get("max_depth")
        if bw is not None and md is not None:
            cfg = GeneratorConfig(beam_width=bw, max_depth=md, top_k_results=10)
            cfg_name = f"bw{bw}_md{md}"
            combinations.append((cfg_name, cfg, profile_name, prof, source))
        else:
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

            if not results:
                logger.warning(f"Sin trayectorias generadas para {experiment_id}. Saltando metricas.")
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
            logger.warning(f"Experimento {experiment_id} fallo: {exc}")

    # Guardar resultados
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
    Imprime un resumen ejecutivo. Protegido si metrics esta vacio.
    FIX ENCODING: usa solo ASCII para evitar UnicodeEncodeError en Windows cp1252.
    """
    if not metrics:
        logger.warning("No hay metricas para resumir.")
        return

    def _rich_print(msg: str) -> None:
        if _rich_available and _console:
            _console.print(msg)
        else:
            logger.info(msg.replace("[bold cyan]","").replace("[/bold cyan]","")
                          .replace("[green]","").replace("[/green]","")
                          .replace("[yellow]","").replace("[/yellow]",""))

    _rich_print("\n[bold cyan]=== RESUMEN EXPERIMENTAL ===[/bold cyan]")
    _rich_print(f"  Experimentos completados: [green]{len(metrics)}[/green]")

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
        f"  Mas rapido          : [yellow]{fastest.config_name}[/yellow] "
        f"({fastest.execution_time_ms:.1f}ms)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathForge -- Diseno Experimental")

    parser.add_argument(
        "--domain", type=str, default=None,
        help="ID del domain graph a usar (ej: software_development). "
             "Por defecto usa careers.json."
    )
    parser.add_argument(
        "--instances", type=str, default=None,
        help="Ruta al archivo JSON con instancias personalizadas."
    )

    args = parser.parse_args()

    inst_path = Path(args.instances) if args.instances else None

    run_all_experiments(domain_id=args.domain, instances_path=inst_path)
