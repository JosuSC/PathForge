"""
experiments/run_experiments.py
-------------------------------
Diseno experimental completo de PathForge.

Ejecuta experimentos SOBRE TODOS LOS DOMINIOS en backend/data/problems/.
(NO usa careers.json — solo problemas de dominio real.)

Para cada dominio compara:
    - 4 configuraciones del generador (narrow_shallow, narrow_deep, wide_shallow, wide_deep)
    - 4 perfiles de restricciones (conservative, ambitious, balanced, fast_track)
    - Multiples nodos de inicio (de instances.json o auto-detectados)
    - Simulacion Monte Carlo de las mejores trayectorias

Produce:
    - Salida en consola Rica y explicativa (Rich)
    - Simulacion estocastica Monte Carlo por cada combinacion
    - Analisis cualitativo LLM (comparacion entre configuraciones)
    - Comparacion cuantitativa entre dominios
    - CSV + JSON con todos los resultados
    - Analisis del comportamiento del sistema

Uso:
    python -m backend.experiments.run_experiments
    python -m backend.experiments.run_experiments --domain "name_of_domain"
    python -m backend.experiments.run_experiments --no-llm
    python -m backend.experiments.run_experiments --quick
"""

from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

# Bootstrap sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich.rule import Rule
from rich.markdown import Markdown

from backend.core.generator import TrajectoryGenerator, GeneratorConfig
from backend.core.graph import CareerGraph
from backend.core.simulation import CareerSimulator
from backend.data.loader import load_domain_graph, list_available_domains
from backend.core.constraints import ConstraintProfiles, Constraint
from backend.experiments.metrics import ExperimentMetrics, compute_metrics

# Intentar importar LLM (no rompe si no esta configurado)
try:
    from backend.llm.analyzer import TrajectoryAnalyzer
    from backend.llm.client import get_llm_client
    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

LLM_TIMEOUT_SECS: int = 90
N_MONTE_CARLO_SIMS: int = 30
RESULTS_DIR = Path(__file__).resolve().parent / "results"

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


def _fresh_profiles() -> dict[str, Constraint]:
    """Crea perfiles de restricciones NUEVOS para cada dominio.

    FIX CACHE: Las PercentileConstraint cachean _threshold por graph_id,
    pero es mas seguro crear objetos frescos para evitar cualquier
    contaminacion cruzada entre dominios.
    """
    return {
        "conservative": ConstraintProfiles.conservative(),
        "ambitious":    ConstraintProfiles.ambitious(),
        "balanced":     ConstraintProfiles.balanced(),
        "fast_track":   ConstraintProfiles.fast_track(),
    }


console = Console()


# ---------------------------------------------------------------------------
# Estructura de resultado
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Resultado por instancia x configuracion."""
    domain:              str
    instance_id:         str
    source_career:       str
    profile:             str
    config_name:         str
    beam_width:          int
    max_depth:           int

    # Estado
    feasible:            bool  = False
    computation_time_ms: float = 0.0

    # Conteos
    num_paths_found:     int   = 0
    pareto_front_size:   int   = 0
    pareto_rank_of_best: int   = -1

    # Objetivos Pareto (mejor trayectoria)
    best_salary_growth:  float = 0.0
    best_final_salary:   float = 0.0
    best_satisfaction:   float = 0.0
    best_avg_demand:     float = 0.0
    best_transition_prob: float = 0.0
    avg_risk:            float = 0.0
    avg_years:           float = 0.0
    terminal_rate:       float = 0.0

    # Diversidad
    diversity_score:     float = 0.0

    # Simulacion Monte Carlo
    sim_success_score_mean: float = 0.0
    sim_success_score_std:  float = 0.0
    sim_salary_p50:         float = 0.0
    sim_salary_p90:         float = 0.0
    sim_years_mean:         float = 0.0
    sim_risk_event_rate:    float = 0.0

    # LLM
    llm_provider:        str   = ""
    llm_analysis_snippet: str  = ""


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def _extract_result(
    domain:         str,
    instance_id:    str,
    source:         str,
    profile_name:   str,
    config_name:    str,
    beam_width:     int,
    max_depth:      int,
    results:        list,
    elapsed_ms:     float,
    metrics:        ExperimentMetrics,
    sim_data:       dict | None = None,
    llm_result=None,
) -> ExperimentResult:
    feasible = len(results) > 0
    best = results[0] if feasible else None
    bs = best.scores if best else {}

    return ExperimentResult(
        domain=domain,
        instance_id=instance_id,
        source_career=source,
        profile=profile_name,
        config_name=config_name,
        beam_width=beam_width,
        max_depth=max_depth,
        feasible=feasible,
        computation_time_ms=round(elapsed_ms, 2),
        num_paths_found=len(results),
        pareto_front_size=metrics.pareto_front_size,
        pareto_rank_of_best=best.pareto_rank if best else -1,
        best_salary_growth=round(bs.get("salary_growth", 0.0), 4),
        best_final_salary=round(bs.get("final_salary", 0.0), 2),
        best_satisfaction=round(bs.get("avg_satisfaction", 0.0), 4),
        best_avg_demand=round(bs.get("avg_demand", 0.0), 4),
        best_transition_prob=round(bs.get("transition_probability_score", 0.0), 4),
        avg_risk=round(metrics.avg_risk, 4),
        avg_years=round(metrics.avg_years, 2),
        terminal_rate=round(metrics.terminal_rate, 4),
        diversity_score=round(metrics.diversity_score, 4),
        sim_success_score_mean=round(sim_data.get("success_score", {}).get("mean", 0.0), 3) if sim_data else 0.0,
        sim_success_score_std=round(sim_data.get("success_score", {}).get("std", 0.0), 3) if sim_data else 0.0,
        sim_salary_p50=round(sim_data.get("salary", {}).get("p50", 0.0), 0) if sim_data else 0.0,
        sim_salary_p90=round(sim_data.get("salary", {}).get("p90", 0.0), 0) if sim_data else 0.0,
        sim_years_mean=round(sim_data.get("years", {}).get("mean", 0.0), 1) if sim_data else 0.0,
        sim_risk_event_rate=round(sum(sim_data.get("risk_events", {}).values()), 2) if sim_data else 0.0,
        llm_provider=llm_result.provider_used if llm_result else "",
        llm_analysis_snippet=(llm_result.content[:800] if llm_result else ""),
    )


def _load_graph_and_sources(
    domain_id: str,
    instances_path: Path | None = None,
) -> tuple[CareerGraph, list[dict], dict]:
    """Carga el grafo del dominio y retorna (graph, instances, metadata)."""
    metadata: dict[str, Any] = {}

    raw_graph = load_domain_graph(domain_id)
    # NO usamos predictor (scorer eliminado del flujo)
    graph = CareerGraph(raw_graph)

    # Cargar metadata
    meta_path = Path(__file__).resolve().parent.parent / "data" / "problems" / domain_id / "metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not metadata:
        metadata = {
            "id": domain_id,
            "domain": domain_id,
            "num_nodes": graph._g.number_of_nodes(),
            "num_edges": graph._g.number_of_edges(),
        }

    # Cargar instancias desde archivo
    if instances_path is not None and instances_path.exists():
        instances = json.loads(instances_path.read_text(encoding="utf-8"))
        return graph, instances, metadata

    # Auto-generar instancias
    source_nodes = [
        nid for nid in graph.all_node_ids()
        if graph.node_attrs(nid).get("type") in ("entry", "mid")
    ]
    if not source_nodes:
        source_nodes = graph.all_node_ids()

    # Priorizar los que tienen mas sucesores
    source_nodes.sort(key=lambda nid: len(graph.successors(nid)), reverse=True)
    source_nodes = source_nodes[:4]

    auto_instances = []
    for source in source_nodes:
        for profile_name in CONSTRAINT_PROFILES:
            for cfg_name, cfg in GENERATOR_CONFIGS.items():
                auto_instances.append({
                    "id": f"auto_{source}_{profile_name}_{cfg_name}",
                    "source_career": source,
                    "profile": profile_name,
                    "config_name": cfg_name,
                    "beam_width": cfg.beam_width,
                    "max_depth": cfg.max_depth,
                })

    return graph, auto_instances, metadata


def _run_monte_carlo(
    graph: CareerGraph,
    path: tuple[str, ...],
    max_salary_ref: float,
) -> dict:
    """Ejecuta simulacion Monte Carlo sobre la mejor trayectoria."""
    if not path or len(path) < 2:
        return {}

    # Construir diccionarios de atributos para el simulador
    node_attrs = {nid: graph.node_attrs(nid) for nid in graph.all_node_ids()}
    edge_attrs = {}
    for nid in graph.all_node_ids():
        for succ in graph.successors(nid):
            edge_attrs[(nid, succ)] = graph.edge_attrs(nid, succ)

    simulator = CareerSimulator(
        seed=42,
        n_simulations=N_MONTE_CARLO_SIMS,
        max_salary_ref=max_salary_ref,
    )

    try:
        return simulator.monte_carlo(path, node_attrs, edge_attrs)
    except Exception as e:
        logger.warning(f"Monte Carlo fallo: {e}")
        return {}


# ---------------------------------------------------------------------------
# Banner y presentacion
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    console.print()
    console.print(Rule(style="bold cyan"))
    console.print(Text.from_markup(
        "\n  [bold cyan]PathForge[/bold cyan] [bold white]— Diseno Experimental Completo[/bold white]\n"
        "  [dim]Experimentos sobre dominios profesionales con simulacion Monte Carlo[/dim]\n"
        "  [dim]y analisis comparativo de configuraciones Beam Search[/dim]\n"
    ))
    console.print(Rule(style="bold cyan"))
    console.print()


def _print_domain_header(domain_id: str, metadata: dict, graph: CareerGraph, n_instances: int) -> None:
    sector = metadata.get("broad_sector", "N/A")
    n_nodes = graph._g.number_of_nodes()
    n_edges = graph._g.number_of_edges()
    terminals = graph.terminal_nodes()
    salaries = [graph.node_attrs(nid).get("avg_salary", 0) for nid in graph.all_node_ids()]
    min_sal = min(salaries) if salaries else 0
    max_sal = max(salaries) if salaries else 0

    console.print()
    console.print(Rule(f"[bold yellow]{domain_id.upper()}[/bold yellow]", style="yellow"))
    console.print(Panel(
        f"[bold]Sector:[/bold] {sector}  |  "
        f"[bold]Nodos:[/bold] {n_nodes}  |  "
        f"[bold]Aristas:[/bold] {n_edges}  |  "
        f"[bold]Terminales:[/bold] {len(terminals)}\n"
        f"[bold]Rango salarial:[/bold] ${min_sal:,.0f} — ${max_sal:,.0f}  |  "
        f"[bold]Instancias:[/bold] {n_instances}  |  "
        f"[bold]Configuraciones:[/bold] {len(GENERATOR_CONFIGS)} x {len(CONSTRAINT_PROFILES)} perfiles",
        title=f"[bold cyan]Dominio: {domain_id}[/bold cyan]",
        border_style="cyan",
    ))


def _print_config_table() -> None:
    table = Table(title="Configuraciones del Generador", show_header=True, header_style="bold magenta")
    table.add_column("Nombre", style="cyan", width=18)
    table.add_column("Beam Width", justify="right", style="green")
    table.add_column("Max Depth", justify="right", style="green")
    table.add_column("Descripcion", style="dim")

    descs = {
        "narrow_shallow": "Exploracion limitada, caminos cortos",
        "narrow_deep":    "Exploracion limitada, caminos largos",
        "wide_shallow":   "Exploracion amplia, caminos cortos",
        "wide_deep":      "Exploracion amplia, caminos largos",
    }

    for name, cfg in GENERATOR_CONFIGS.items():
        table.add_row(name, str(cfg.beam_width), str(cfg.max_depth), descs.get(name, ""))
    console.print(table)

    table2 = Table(title="Perfiles de Restricciones", show_header=True, header_style="bold magenta")
    table2.add_column("Perfil", style="cyan", width=15)
    table2.add_column("Restricciones", style="green")

    profile_descs = {
        "conservative": "MaxRisk(p65) AND MaxDifficulty(p75)",
        "ambitious":    "MinSalary(p25) AND MinLength(2)",
        "balanced":     "MaxYears(12) AND MaxRisk(p75) AND MinSalary(p20)",
        "fast_track":   "MaxYears(6) AND MinSalary(p35)",
    }
    for name in CONSTRAINT_PROFILES:
        table2.add_row(name, profile_descs.get(name, str(CONSTRAINT_PROFILES[name])))
    console.print(table2)
    console.print()


# ---------------------------------------------------------------------------
# Ejecucion de un dominio
# ---------------------------------------------------------------------------

def _run_domain_experiments(
    domain_id: str,
    use_llm: bool = True,
    quick: bool = False,
) -> list[ExperimentResult]:
    """Ejecuta todos los experimentos para un dominio y retorna resultados."""

    # FIX CACHE: perfiles frescos para cada dominio
    fresh_profiles = _fresh_profiles()

    # Determinar ruta de instancias
    instances_path = None
    inst_path = Path(__file__).resolve().parent.parent / "data" / "problems" / domain_id / "instances.json"
    if inst_path.exists():
        instances_path = inst_path

    graph, instances, metadata = _load_graph_and_sources(domain_id, instances_path)

    # Quick mode: limitar instancias
    if quick and len(instances) > 16:
        instances = instances[:16]

    _print_domain_header(domain_id, metadata, graph, len(instances))

    # Calcular max_salary_ref para simulacion
    salaries = [graph.node_attrs(nid).get("avg_salary", 0) for nid in graph.all_node_ids()]
    max_salary_ref = max(salaries) if salaries else 100_000

    # Inicializar LLM
    llm_analyzer = None
    if use_llm and _LLM_AVAILABLE:
        try:
            llm_client = get_llm_client()
            llm_analyzer = TrajectoryAnalyzer(client=llm_client)
            console.print("[green]  LLM disponible — se ejecutara analisis cualitativo.[/green]")
        except Exception as e:
            console.print(f"[yellow]  LLM no disponible: {e}[/yellow]")
            llm_analyzer = None

    console.print(f"[dim]  Simulacion Monte Carlo: {N_MONTE_CARLO_SIMS} iteraciones por trayectoria[/dim]")

    all_results: list[ExperimentResult] = []
    combinations = _build_combinations(instances, graph, fresh_profiles)

    console.print(f"\n[bold]Ejecutando {len(combinations)} combinaciones experimentales...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]{domain_id}[/cyan]", total=len(combinations))

        for combo in combinations:
            cfg_name = combo["config_name"]
            cfg = combo["config"]
            prof_name = combo["profile_name"]
            prof = combo["profile"]
            source = combo["source"]
            inst_id = combo["instance_id"]

            progress.update(task, description=f"[cyan]{domain_id}[/cyan] | {cfg_name} | {prof_name}")

            try:
                if source not in graph.all_node_ids():
                    progress.advance(task)
                    continue

                generator = TrajectoryGenerator(graph, cfg)
                start = time.perf_counter()
                results = generator.generate(source=source, constraints=prof)
                elapsed_ms = (time.perf_counter() - start) * 1000

                if not results:
                    metrics = compute_metrics([], cfg_name, source, prof_name, elapsed_ms)
                    exp_result = _extract_result(
                        domain=domain_id, instance_id=inst_id,
                        source=source, profile_name=prof_name, config_name=cfg_name,
                        beam_width=cfg.beam_width, max_depth=cfg.max_depth,
                        results=[], elapsed_ms=elapsed_ms, metrics=metrics,
                    )
                    all_results.append(exp_result)
                    progress.advance(task)
                    continue

                metrics = compute_metrics(results, cfg_name, source, prof_name, elapsed_ms)

                # Simulacion Monte Carlo sobre la mejor trayectoria
                sim_data = None
                try:
                    best_path = results[0].trajectory.nodes
                    sim_data = _run_monte_carlo(graph, best_path, max_salary_ref)
                except Exception as e:
                    logger.debug(f"Sim fallo para {inst_id}: {e}")

                # LLM solo para la primera instancia de cada (config, profile) combo
                llm_result = None
                if llm_analyzer and results and _should_run_llm(all_results, cfg_name, prof_name):
                    try:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                llm_analyzer.rank_by,
                                results,
                                "mayor potencial de crecimiento profesional a largo plazo",
                                min(5, len(results)),
                            )
                            llm_result = future.result(timeout=LLM_TIMEOUT_SECS)
                    except FuturesTimeoutError:
                        logger.warning(f"LLM timeout ({LLM_TIMEOUT_SECS}s)")
                    except Exception as e:
                        logger.warning(f"LLM error: {e}")

                exp_result = _extract_result(
                    domain=domain_id, instance_id=inst_id,
                    source=source, profile_name=prof_name, config_name=cfg_name,
                    beam_width=cfg.beam_width, max_depth=cfg.max_depth,
                    results=results, elapsed_ms=elapsed_ms, metrics=metrics,
                    sim_data=sim_data, llm_result=llm_result,
                )
                all_results.append(exp_result)

            except Exception as exc:
                logger.warning(f"Error en {inst_id}: {exc}")
                metrics = compute_metrics([], cfg_name, source, prof_name, 0)
                exp_result = _extract_result(
                    domain=domain_id, instance_id=inst_id,
                    source=source, profile_name=prof_name, config_name=cfg_name,
                    beam_width=cfg.beam_width, max_depth=cfg.max_depth,
                    results=[], elapsed_ms=0, metrics=metrics,
                )
                all_results.append(exp_result)

            progress.advance(task)

    # Imprimir resultados del dominio
    _print_domain_results(domain_id, all_results)

    return all_results


# Mapeo inverso: (beam_width, max_depth) -> config_name
_BW_MD_TO_NAME: dict[tuple[int, int], str] = {
    (cfg.beam_width, cfg.max_depth): name
    for name, cfg in GENERATOR_CONFIGS.items()
}


def _build_combinations(
    instances: list[dict],
    graph: CareerGraph,
    profiles: dict[str, Constraint] | None = None,
) -> list[dict]:
    """Construye la lista de combinaciones a ejecutar."""
    combinations = []
    active_profiles = profiles if profiles is not None else CONSTRAINT_PROFILES
    default_profile = ConstraintProfiles.balanced()

    for inst in instances:
        source = inst["source_career"]
        profile_name = inst.get("profile", "balanced")
        prof = active_profiles.get(profile_name, default_profile)

        bw = inst.get("beam_width")
        md = inst.get("max_depth")
        explicit_name = inst.get("config_name")

        if bw is not None and md is not None:
            cfg = GeneratorConfig(beam_width=bw, max_depth=md, top_k_results=10)
            if explicit_name:
                cfg_name = explicit_name
            else:
                cfg_name = _BW_MD_TO_NAME.get((bw, md), f"bw{bw}_md{md}")
            combinations.append({
                "config_name": cfg_name, "config": cfg,
                "profile_name": profile_name, "profile": prof,
                "source": source, "instance_id": inst.get("id", f"auto_{source}"),
            })
        else:
            for cfg_name, cfg in GENERATOR_CONFIGS.items():
                combinations.append({
                    "config_name": cfg_name, "config": cfg,
                    "profile_name": profile_name, "profile": prof,
                    "source": source, "instance_id": inst.get("id", f"auto_{source}"),
                })

    return combinations


def _should_run_llm(existing: list[ExperimentResult], cfg_name: str, prof_name: str) -> bool:
    """Ejecutar LLM solo una vez por combinacion (config, profile)."""
    for r in existing:
        if r.config_name == cfg_name and r.profile == prof_name and r.llm_analysis_snippet:
            return False
    return True


# ---------------------------------------------------------------------------
# Presentacion de resultados por dominio
# ---------------------------------------------------------------------------

def _print_domain_results(domain_id: str, results: list[ExperimentResult]) -> None:
    """Imprime la tabla de resultados y resumen para un dominio."""
    domain_results = [r for r in results if r.domain == domain_id]
    if not domain_results:
        console.print(f"[yellow]Sin resultados para {domain_id}[/yellow]")
        return

    feasible = [r for r in domain_results if r.feasible]
    infeasible = [r for r in domain_results if not r.feasible]

    table = Table(
        title=f"Resultados — {domain_id}",
        show_header=True, header_style="bold magenta",
        caption=f"{len(feasible)} factibles / {len(infeasible)} infactibles de {len(domain_results)} total"
    )
    table.add_column("Config", style="cyan", width=16)
    table.add_column("Perfil", style="yellow", width=13)
    table.add_column("Source", style="dim", width=16)
    table.add_column("Tray.", justify="right", style="green")
    table.add_column("Pareto", justify="right")
    table.add_column("Div.", justify="right", style="magenta")
    table.add_column("Salary Gr.", justify="right", style="green")
    table.add_column("Risk", justify="right", style="red")
    table.add_column("Sim Score", justify="right", style="blue")
    table.add_column("T.Rate", justify="right", style="blue")
    table.add_column("Time(ms)", justify="right", style="dim")

    for r in feasible:
        table.add_row(
            r.config_name,
            r.profile,
            r.source_career[:15],
            str(r.num_paths_found),
            str(r.pareto_front_size),
            f"{r.diversity_score:.3f}",
            f"{r.best_salary_growth:.1%}",
            f"{r.avg_risk:.2f}",
            f"{r.sim_success_score_mean:.3f}",
            f"{r.terminal_rate:.2f}",
            f"{r.computation_time_ms:.0f}",
        )

    console.print(table)

    if infeasible:
        console.print(f"[dim]  {len(infeasible)} combinaciones no produjeron trayectorias factibles.[/dim]")


# ---------------------------------------------------------------------------
# Analisis comparativo entre configuraciones
# ---------------------------------------------------------------------------

def _print_config_comparison(results: list[ExperimentResult]) -> None:
    console.print()
    console.print(Rule("[bold cyan]Comparacion de Configuraciones del Generador[/bold cyan]", style="cyan"))

    table = Table(show_header=True, header_style="bold magenta", title="Rendimiento por Configuracion (promedio)")
    table.add_column("Configuracion", style="cyan", width=18)
    table.add_column("Trayect. (avg)", justify="right", style="green")
    table.add_column("Factibles", justify="right")
    table.add_column("Diversidad", justify="right", style="magenta")
    table.add_column("Salary Gr.", justify="right", style="green")
    table.add_column("Sim Score", justify="right", style="blue")
    table.add_column("Risk (avg)", justify="right", style="red")
    table.add_column("Tiempo (ms)", justify="right", style="dim")

    for cfg_name in GENERATOR_CONFIGS:
        cfg_results = [r for r in results if r.config_name == cfg_name and r.feasible]
        if not cfg_results:
            table.add_row(cfg_name, "—", "0", "—", "—", "—", "—", "—")
            continue

        n = len(cfg_results)
        avg_paths = sum(r.num_paths_found for r in cfg_results) / n
        avg_div = sum(r.diversity_score for r in cfg_results) / n
        avg_sgr = sum(r.best_salary_growth for r in cfg_results) / n
        avg_sim = sum(r.sim_success_score_mean for r in cfg_results) / n
        avg_risk = sum(r.avg_risk for r in cfg_results) / n
        avg_time = sum(r.computation_time_ms for r in cfg_results) / n

        table.add_row(
            cfg_name,
            f"{avg_paths:.1f}",
            str(n),
            f"{avg_div:.3f}",
            f"{avg_sgr:.1%}",
            f"{avg_sim:.3f}",
            f"{avg_risk:.3f}",
            f"{avg_time:.0f}",
        )

    console.print(table)


def _print_profile_comparison(results: list[ExperimentResult]) -> None:
    console.print()
    console.print(Rule("[bold cyan]Comparacion de Perfiles de Restricciones[/bold cyan]", style="cyan"))

    table = Table(show_header=True, header_style="bold magenta", title="Rendimiento por Perfil (promedio)")
    table.add_column("Perfil", style="yellow", width=15)
    table.add_column("Factibles", justify="right")
    table.add_column("Infactibles", justify="right", style="red")
    table.add_column("Tasa Exito", justify="right", style="green")
    table.add_column("Diversidad", justify="right", style="magenta")
    table.add_column("Sim Score", justify="right", style="blue")
    table.add_column("Salary Gr.", justify="right", style="green")
    table.add_column("Risk (avg)", justify="right", style="red")

    for prof_name in CONSTRAINT_PROFILES:
        prof_results = [r for r in results if r.profile == prof_name]
        feasible = [r for r in prof_results if r.feasible]
        infeasible = [r for r in prof_results if not r.feasible]
        total = len(prof_results)

        if total == 0:
            table.add_row(prof_name, "0", "0", "—", "—", "—", "—", "—")
            continue

        success_rate = len(feasible) / total
        n_f = len(feasible)

        if n_f > 0:
            avg_div = sum(r.diversity_score for r in feasible) / n_f
            avg_sim = sum(r.sim_success_score_mean for r in feasible) / n_f
            avg_sgr = sum(r.best_salary_growth for r in feasible) / n_f
            avg_risk = sum(r.avg_risk for r in feasible) / n_f
        else:
            avg_div = avg_sim = avg_sgr = avg_risk = 0

        table.add_row(
            prof_name,
            str(n_f),
            str(len(infeasible)),
            f"{success_rate:.0%}",
            f"{avg_div:.3f}",
            f"{avg_sim:.3f}",
            f"{avg_sgr:.1%}",
            f"{avg_risk:.3f}",
        )

    console.print(table)


def _print_domain_comparison(all_results: list[ExperimentResult]) -> None:
    console.print()
    console.print(Rule("[bold cyan]Comparacion entre Dominios[/bold cyan]", style="cyan"))

    domains = sorted(set(r.domain for r in all_results))

    table = Table(show_header=True, header_style="bold magenta", title="Rendimiento por Dominio")
    table.add_column("Dominio", style="cyan", width=22)
    table.add_column("Factibles", justify="right", style="green")
    table.add_column("Total", justify="right")
    table.add_column("Tasa Exito", justify="right", style="bold green")
    table.add_column("Diversidad", justify="right", style="magenta")
    table.add_column("Sim Score", justify="right", style="blue")
    table.add_column("Risk", justify="right", style="red")
    table.add_column("Tiempo (ms)", justify="right", style="dim")

    for domain in domains:
        domain_results = [r for r in all_results if r.domain == domain]
        feasible = [r for r in domain_results if r.feasible]
        total = len(domain_results)
        n_f = len(feasible)

        if total == 0:
            continue

        success_rate = n_f / total

        if n_f > 0:
            avg_div = sum(r.diversity_score for r in feasible) / n_f
            avg_sim = sum(r.sim_success_score_mean for r in feasible) / n_f
            avg_risk = sum(r.avg_risk for r in feasible) / n_f
            avg_time = sum(r.computation_time_ms for r in feasible) / n_f
        else:
            avg_div = avg_sim = avg_risk = avg_time = 0

        rate_style = "bold green" if success_rate >= 0.8 else "yellow" if success_rate >= 0.5 else "red"

        table.add_row(
            domain,
            str(n_f),
            str(total),
            f"[{rate_style}]{success_rate:.0%}[/{rate_style}]",
            f"{avg_div:.3f}",
            f"{avg_sim:.3f}",
            f"{avg_risk:.3f}",
            f"{avg_time:.0f}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Analisis del comportamiento del sistema
# ---------------------------------------------------------------------------

def _print_system_analysis(all_results: list[ExperimentResult]) -> None:
    console.print()
    console.print(Rule("[bold cyan]Analisis del Comportamiento del Sistema[/bold cyan]", style="cyan"))

    feasible = [r for r in all_results if r.feasible]
    infeasible = [r for r in all_results if not r.feasible]
    total = len(all_results)

    if total == 0:
        console.print("[yellow]Sin resultados para analizar.[/yellow]")
        return

    success_rate = len(feasible) / total
    console.print(Panel(
        f"[bold]Tasa de exito global:[/bold] [{'green' if success_rate >= 0.7 else 'yellow' if success_rate >= 0.4 else 'red'}]"
        f"{success_rate:.1%}[/{'green' if success_rate >= 0.7 else 'yellow' if success_rate >= 0.4 else 'red'}]"
        f" ({len(feasible)}/{total} combinaciones factibles)\n\n"
        f"[bold]Causas de infactibilidad:[/bold] {len(infeasible)} combinaciones no generaron trayectorias.\n"
        f"  - Restricciones muy estrictas para ciertos dominios (ej: MinSalary alto en salarios bajos)\n"
        f"  - Nodos de inicio con pocos sucesores bajo el perfil seleccionado\n"
        f"  - Perfiles como fast_track (MaxYears=6) son inherentemente restrictivos",
        title="[bold]1. Viabilidad del Sistema[/bold]",
        border_style="green",
    ))

    if not feasible:
        return

    # 2. Efecto del beam width y profundidad
    console.print(Panel(
        _analyze_beam_effects(feasible),
        title="[bold]2. Efecto del Beam Width y Profundidad[/bold]",
        border_style="yellow",
    ))

    # 3. Sensibilidad a restricciones
    console.print(Panel(
        _analyze_constraint_sensitivity(all_results),
        title="[bold]3. Sensibilidad a Restricciones[/bold]",
        border_style="red",
    ))

    # 4. Simulacion Monte Carlo
    console.print(Panel(
        _analyze_simulation(feasible),
        title="[bold]4. Analisis de Simulacion Monte Carlo[/bold]",
        border_style="blue",
    ))

    # 5. Mejores configuraciones por dominio
    console.print(Panel(
        _analyze_best_configs(feasible),
        title="[bold]5. Mejores Configuraciones por Dominio[/bold]",
        border_style="cyan",
    ))


def _analyze_beam_effects(results: list[ExperimentResult]) -> str:
    lines = []
    for cfg_name in GENERATOR_CONFIGS:
        cfg_results = [r for r in results if r.config_name == cfg_name]
        if not cfg_results:
            continue
        n = len(cfg_results)
        avg_paths = sum(r.num_paths_found for r in cfg_results) / n
        avg_pareto = sum(r.pareto_front_size for r in cfg_results) / n
        avg_time = sum(r.computation_time_ms for r in cfg_results) / n
        lines.append(
            f"  [cyan]{cfg_name:18s}[/cyan] | "
            f"trayect={avg_paths:5.1f} | "
            f"pareto={avg_pareto:4.1f} | "
            f"tiempo={avg_time:7.0f}ms"
        )

    shallow = [r for r in results if "shallow" in r.config_name]
    deep = [r for r in results if "deep" in r.config_name]
    if shallow and deep:
        shallow_time = sum(r.computation_time_ms for r in shallow) / len(shallow)
        deep_time = sum(r.computation_time_ms for r in deep) / len(deep)
        if shallow_time > 0:
            lines.append(f"\n[green]Hallazgo:[/green] deep (depth=6) tarda {deep_time/shallow_time:.1f}x mas que shallow (depth=3)")
        lines.append("[dim]Mayor profundidad explora caminos mas largos pero incrementa el tiempo cuadraticamente con el beam width.[/dim]")

    return "\n".join(lines)


def _analyze_constraint_sensitivity(results: list[ExperimentResult]) -> str:
    lines = []
    for prof_name in CONSTRAINT_PROFILES:
        prof_results = [r for r in results if r.profile == prof_name]
        total = len(prof_results)
        feasible = [r for r in prof_results if r.feasible]
        if total == 0:
            continue
        rate = len(feasible) / total
        lines.append(
            f"  [yellow]{prof_name:15s}[/yellow] | "
            f"factibles={len(feasible):3d}/{total} ({rate:.0%})"
        )

    rates = {}
    for prof_name in CONSTRAINT_PROFILES:
        prof_results = [r for r in results if r.profile == prof_name]
        if prof_results:
            rates[prof_name] = len([r for r in prof_results if r.feasible]) / len(prof_results)

    if rates:
        most_restrictive = min(rates, key=rates.get)
        most_permissive = max(rates, key=rates.get)
        lines.append(f"\n[red]Mas restrictivo:[/red] {most_restrictive} (tasa exito {rates[most_restrictive]:.0%})")
        lines.append(f"[green]Mas permisivo:[/green] {most_permissive} (tasa exito {rates[most_permissive]:.0%})")
        lines.append("[dim]Las restricciones adaptativas por percentil ajustan automaticamente los umbrales al rango de datos de cada dominio.[/dim]")

    return "\n".join(lines)


def _analyze_simulation(results: list[ExperimentResult]) -> str:
    lines = []
    # Promedios de simulacion por perfil
    for prof_name in CONSTRAINT_PROFILES:
        prof_results = [r for r in results if r.profile == prof_name and r.sim_success_score_mean > 0]
        if not prof_results:
            continue
        n = len(prof_results)
        avg_sim = sum(r.sim_success_score_mean for r in prof_results) / n
        avg_std = sum(r.sim_success_score_std for r in prof_results) / n
        avg_risk_ev = sum(r.sim_risk_event_rate for r in prof_results) / n
        lines.append(
            f"  [yellow]{prof_name:15s}[/yellow] | "
            f"score={avg_sim:.3f} +/- {avg_std:.3f} | "
            f"eventos_riesgo={avg_risk_ev:.2f}"
        )

    lines.append("\n[dim]La simulacion Monte Carlo (30 iteraciones) modela eventos estocasticos como promociones, despidos, "
                 "burnout y crisis de mercado. El success_score combina salario normalizado, satisfaccion, "
                 "tiempo efectivo y proporcion de eventos negativos.[/dim]")

    # Mejor y peor caso de simulacion
    with_sim = [r for r in results if r.sim_success_score_mean > 0]
    if with_sim:
        best_sim = max(with_sim, key=lambda r: r.sim_success_score_mean)
        worst_sim = min(with_sim, key=lambda r: r.sim_success_score_mean)
        lines.append(f"\n[green]Mejor simulacion:[/green] {best_sim.domain}/{best_sim.config_name}/{best_sim.profile} "
                     f"(score={best_sim.sim_success_score_mean:.3f})")
        lines.append(f"[red]Peor simulacion:[/red] {worst_sim.domain}/{worst_sim.config_name}/{worst_sim.profile} "
                     f"(score={worst_sim.sim_success_score_mean:.3f})")

    return "\n".join(lines)


def _analyze_best_configs(results: list[ExperimentResult]) -> str:
    lines = []
    domains = sorted(set(r.domain for r in results))

    for domain in domains:
        domain_results = [r for r in results if r.domain == domain]
        if not domain_results:
            continue

        best_div = max(domain_results, key=lambda r: r.diversity_score)
        best_sim = max(domain_results, key=lambda r: r.sim_success_score_mean)
        feasible = [r for r in domain_results if r.feasible]
        best_time = min(feasible, key=lambda r: r.computation_time_ms) if feasible else None

        lines.append(f"[bold cyan]{domain}[/bold cyan]")
        lines.append(f"  Mejor diversidad:  {best_div.config_name} + {best_div.profile} (div={best_div.diversity_score:.3f})")
        lines.append(f"  Mejor sim. score:  {best_sim.config_name} + {best_sim.profile} (score={best_sim.sim_success_score_mean:.3f})")
        if best_time:
            lines.append(f"  Mas rapido:        {best_time.config_name} + {best_time.profile} ({best_time.computation_time_ms:.0f}ms)")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analisis LLM
# ---------------------------------------------------------------------------

def _print_llm_analysis(all_results: list[ExperimentResult]) -> None:
    llm_results = [r for r in all_results if r.llm_analysis_snippet]
    if not llm_results:
        console.print("\n[dim]No se obtuvieron analisis LLM.[/dim]")
        return

    console.print()
    console.print(Rule("[bold magenta]Analisis Cualitativo del LLM[/bold magenta]", style="magenta"))

    by_domain: dict[str, list[ExperimentResult]] = {}
    for r in llm_results:
        by_domain.setdefault(r.domain, []).append(r)

    for domain, domain_llm in by_domain.items():
        console.print(f"\n[bold cyan]Dominio: {domain}[/bold cyan]")
        for r in domain_llm[:4]:
            console.print(Panel(
                f"[dim]{r.llm_analysis_snippet}[/dim]",
                title=f"[bold]{r.config_name} + {r.profile}[/bold] (via {r.llm_provider})",
                border_style="magenta",
            ))


# ---------------------------------------------------------------------------
# Resumen ejecutivo final
# ---------------------------------------------------------------------------

def _print_executive_summary(all_results: list[ExperimentResult]) -> None:
    console.print()
    console.print(Rule("[bold green]RESUMEN EJECUTIVO[/bold green]", style="green"))

    total = len(all_results)
    feasible = [r for r in all_results if r.feasible]
    infeasible = [r for r in all_results if not r.feasible]
    domains = sorted(set(r.domain for r in all_results))

    if total == 0:
        console.print("[red]No se ejecutaron experimentos.[/red]")
        return

    n = len(feasible)
    avg_div = sum(r.diversity_score for r in feasible) / n if n else 0
    avg_sgr = sum(r.best_salary_growth for r in feasible) / n if n else 0
    avg_sim = sum(r.sim_success_score_mean for r in feasible) / n if n else 0
    avg_risk = sum(r.avg_risk for r in feasible) / n if n else 0
    avg_time = sum(r.computation_time_ms for r in feasible) / n if n else 0

    console.print(Panel(
        f"[bold]Total de combinaciones ejecutadas:[/bold] {total}\n"
        f"[bold]Factibles:[/bold] [green]{len(feasible)}[/green] ({len(feasible)/total:.0%})  |  "
        f"[bold]Infactibles:[/bold] [red]{len(infeasible)}[/red] ({len(infeasible)/total:.0%})\n"
        f"[bold]Dominios evaluados:[/bold] {len(domains)}\n"
        f"[bold]Configuraciones probadas:[/bold] {len(GENERATOR_CONFIGS)} x {len(CONSTRAINT_PROFILES)} = "
        f"{len(GENERATOR_CONFIGS) * len(CONSTRAINT_PROFILES)} combinaciones algoritmicas\n\n"
        f"[bold]Metricas clave (promedio sobre factibles):[/bold]\n"
        f"  Diversidad:          {avg_div:.3f}\n"
        f"  Salary Growth:       {avg_sgr:.1%}\n"
        f"  Sim. Score (MC):     {avg_sim:.3f}\n"
        f"  Riesgo promedio:     {avg_risk:.3f}\n"
        f"  Tiempo promedio:     {avg_time:.0f}ms",
        title="[bold]PathForge — Resultados Experimentales[/bold]",
        border_style="green",
    ))

    if feasible:
        console.print("\n[bold green]Top-3 Mejores por Simulacion Monte Carlo:[/bold green]")
        top3 = sorted(feasible, key=lambda r: r.sim_success_score_mean, reverse=True)[:3]
        for i, r in enumerate(top3, 1):
            console.print(f"  {i}. [cyan]{r.config_name}[/cyan] + [yellow]{r.profile}[/yellow] "
                         f"en [bold]{r.domain}[/bold] — sim_score={r.sim_success_score_mean:.3f}")


# ---------------------------------------------------------------------------
# Guardar resultados
# ---------------------------------------------------------------------------

def _save_results(all_results: list[ExperimentResult]) -> None:
    if not all_results:
        console.print("[yellow]No hay resultados para guardar.[/yellow]")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = RESULTS_DIR / "full_experiment_results.csv"
    fieldnames = list(asdict(all_results[0]).keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in all_results:
            writer.writerow(asdict(res))

    # JSON
    json_path = RESULTS_DIR / "full_experiment_results.json"
    json_data = [asdict(r) for r in all_results]
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Resumen JSON por dominio
    summary: dict[str, Any] = {}
    domains = sorted(set(r.domain for r in all_results))
    for domain in domains:
        domain_results = [r for r in all_results if r.domain == domain]
        feasible = [r for r in domain_results if r.feasible]
        n_f = len(feasible)
        n_total = len(domain_results)
        if n_f > 0:
            summary[domain] = {
                "total_combinations": n_total,
                "feasible": n_f,
                "infeasible": n_total - n_f,
                "success_rate": round(n_f / n_total, 4),
                "avg_diversity": round(sum(r.diversity_score for r in feasible) / n_f, 4),
                "avg_salary_growth": round(sum(r.best_salary_growth for r in feasible) / n_f, 4),
                "avg_sim_score": round(sum(r.sim_success_score_mean for r in feasible) / n_f, 4),
                "avg_risk": round(sum(r.avg_risk for r in feasible) / n_f, 4),
                "avg_terminal_rate": round(sum(r.terminal_rate for r in feasible) / n_f, 4),
                "avg_time_ms": round(sum(r.computation_time_ms for r in feasible) / n_f, 2),
            }
        else:
            summary[domain] = {
                "total_combinations": n_total,
                "feasible": 0,
                "infeasible": n_total,
                "success_rate": 0.0,
            }

    summary_path = RESULTS_DIR / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(f"\n[green]Resultados guardados:[/green]")
    console.print(f"  CSV:     {csv_path}")
    console.print(f"  JSON:    {json_path}")
    console.print(f"  Summary: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_full_experiments(
    domain_id: str | None = None,
    use_llm: bool = True,
    quick: bool = False,
) -> list[ExperimentResult]:
    """Punto de entrada principal."""
    _print_banner()
    _print_config_table()

    all_results: list[ExperimentResult] = []

    # Determinar dominios a ejecutar
    domains_to_run: list[str] = []

    if domain_id:
        domains_to_run = [domain_id]
    elif not quick:
        available = list_available_domains()
        domains_to_run = [d["id"] for d in available]
    else:
        available = list_available_domains()
        if available:
            domains_to_run = [available[len(available) // 2]["id"]]

    console.print(Panel(
        f"[bold]Ejecutando experimentos sobre [cyan]{len(domains_to_run)}[/cyan] dominios[/bold]\n"
        + ", ".join(domains_to_run[:8]) + ("..." if len(domains_to_run) > 8 else ""),
        border_style="yellow",
    ))

    for i, did in enumerate(domains_to_run, 1):
        console.print(f"\n[bold]({i}/{len(domains_to_run)}) Dominio: [yellow]{did}[/yellow][/bold]")
        try:
            domain_results = _run_domain_experiments(domain_id=did, use_llm=use_llm, quick=quick)
            all_results.extend(domain_results)
        except Exception as e:
            console.print(f"[red]Error en dominio {did}: {e}[/red]")
            logger.error(f"Error en dominio {did}: {e}")

    # Analisis comparativo global
    if all_results:
        _print_config_comparison(all_results)
        _print_profile_comparison(all_results)

        if len(set(r.domain for r in all_results)) > 1:
            _print_domain_comparison(all_results)

        _print_system_analysis(all_results)
        _print_llm_analysis(all_results)
        _print_executive_summary(all_results)
        _save_results(all_results)
    else:
        console.print("[bold red]No se obtuvieron resultados de ningun experimento.[/bold red]")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PathForge — Diseno Experimental Completo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m backend.experiments.run_experiments                 # Todos los dominios
  python -m backend.experiments.run_experiments --quick          # 1 dominio representativo
  python -m backend.experiments.run_experiments --domain finance # Solo finance
  python -m backend.experiments.run_experiments --no-llm         # Sin analisis LLM
        """,
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="ID del dominio a ejecutar. Por defecto ejecuta todos."
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Desactiva el analisis LLM cualitativo."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Modo rapido: solo 1 dominio, menos instancias."
    )

    args = parser.parse_args()

    try:
        run_full_experiments(
            domain_id=args.domain,
            use_llm=not args.no_llm,
            quick=args.quick,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Experimentos interrumpidos por el usuario.[/yellow]")
        sys.exit(1)
