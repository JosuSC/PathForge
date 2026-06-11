"""
experiments/run_experiments.py
-------------------------------
Ejecuta el diseño experimental sobre instancias de prueba con comparación
de configuraciones de beam_width, max_depth y evaluación LLM opcional.

"""

from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Bootstrap sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

from backend.core.generator import TrajectoryGenerator, GeneratorConfig
from backend.core.graph import CareerGraph
from backend.core.scorer import CareerOutcomePredictor
from backend.data.loader import load_career_graph
from backend.core.constraints import ConstraintProfiles
from backend.llm.analyzer import TrajectoryAnalyzer
from backend.llm.client import get_llm_client
from backend.experiments.metrics import ExperimentMetrics, compute_metrics

# FIX [RE1]: timeout configurable para cada llamada LLM
LLM_TIMEOUT_SECS: int = 30

# FIX [R4]: mismo directorio de resultados que runner.py
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# FIX [RE2] + [RE3]: estructura de resultado unificada y multiobjetivo
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """
    Resultado por instancia × configuración.

    Hereda conceptualmente de ExperimentMetrics pero añade
    campos específicos del experimento por instancia (instance_id, feasible,
    use_llm, llm_analysis_snippet).
    Captura todos los objetivos Pareto, no solo salary_growth.
    """
    instance_id:         str
    source_career:       str
    profile:             str
    beam_width:          int
    max_depth:           int
    use_llm:             bool

    # Estado
    feasible:            bool  = False
    computation_time_ms: float = 0.0

    # Conteos
    num_paths_found:     int   = 0
    pareto_front_size:   int   = 0
    pareto_rank_of_best: int   = -1

    # FIX [RE2]: todos los objetivos Pareto
    best_salary_growth:  float = 0.0
    best_final_salary:   float = 0.0
    best_satisfaction:   float = 0.0
    best_avg_demand:     float = 0.0
    avg_risk:            float = 0.0
    avg_years:           float = 0.0

    # Diversidad del set generado
    diversity_score:     float = 0.0

    # LLM (solo si use_llm=True)
    llm_provider:        str   = ""
    llm_analysis_snippet: str  = ""  # primeros 200 chars del análisis

    details:             str   = ""


def _extract_result(
    instance_id:    str,
    source:         str,
    profile_name:   str,
    beam_width:     int,
    max_depth:      int,
    use_llm:        bool,
    results:        list,
    elapsed_ms:     float,
    metrics:        ExperimentMetrics,
    llm_result=None,
) -> ExperimentResult:
    """Construye ExperimentResult desde los resultados del generador."""
    feasible = len(results) > 0

    # FIX [RE2]: capturar todos los objetivos del mejor resultado (rank 0)
    best = results[0] if feasible else None
    bs   = best.scores if best else {}

    return ExperimentResult(
        instance_id=instance_id,
        source_career=source,
        profile=profile_name,
        beam_width=beam_width,
        max_depth=max_depth,
        use_llm=use_llm,
        feasible=feasible,
        computation_time_ms=elapsed_ms,
        num_paths_found=len(results),
        pareto_front_size=metrics.pareto_front_size,
        pareto_rank_of_best=best.pareto_rank if best else -1,
        best_salary_growth=round(bs.get("salary_growth",    0.0), 4),
        best_final_salary= round(bs.get("final_salary",     0.0), 2),
        best_satisfaction= round(bs.get("avg_satisfaction", 0.0), 4),
        best_avg_demand=   round(bs.get("avg_demand",       0.0), 4),
        avg_risk=          round(metrics.avg_risk,                4),
        avg_years=         round(metrics.avg_years,               2),
        diversity_score=   round(metrics.diversity_score,         4),
        llm_provider=      llm_result.provider_used if llm_result else "",
        llm_analysis_snippet=(llm_result.content[:200] if llm_result else ""),
    )


# ---------------------------------------------------------------------------
# Experimento principal
# ---------------------------------------------------------------------------

def run_experiments(instances_path: Path, results_path: Path) -> None:
    logger.info("Cargando grafo de carreras...")
    raw_graph    = load_career_graph()
    predictor    = CareerOutcomePredictor.load_or_train()
    career_graph = CareerGraph(raw_graph, outcome_predictor=predictor)

    logger.info(f"Cargando instancias desde {instances_path}...")
    with open(instances_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    instances = test_data["instances"]

    beam_widths_to_test = [3, 5, 7]
    max_depths_to_test  = [4, 6]

    constraint_profiles = {
        "conservative": ConstraintProfiles.conservative(),
        "ambitious":    ConstraintProfiles.ambitious(),
        "balanced":     ConstraintProfiles.balanced(),
        "fast_track":   ConstraintProfiles.fast_track(),
    }

    # LLM opcional
    try:
        llm_client   = get_llm_client()
        llm_analyzer = TrajectoryAnalyzer(client=llm_client)
        llm_available = True
        logger.info("LLM disponible — se ejecutará análisis cualitativo.")
    except Exception as e:
        logger.warning(f"LLM no disponible: {e}. Experimentos sin LLM.")
        llm_available = False
        llm_analyzer  = None

    all_results: list[ExperimentResult] = []
    total_runs = len(instances) * len(beam_widths_to_test) * len(max_depths_to_test)
    logger.info(f"Total de combinaciones algorítmicas: {total_runs}")

    for instance in instances:
        source       = instance["source_career"]
        profile_name = instance["profile"]

        # FIX [RE4]: warning explícito si el perfil no existe
        if profile_name not in constraint_profiles:
            logger.warning(
                f"Perfil '{profile_name}' en instancia '{instance['id']}' "
                f"no está registrado. Usando 'balanced' como fallback. "
                f"Perfiles válidos: {list(constraint_profiles.keys())}"
            )
        constraint = constraint_profiles.get(profile_name, ConstraintProfiles.balanced())

        for beam_width in beam_widths_to_test:
            for max_depth in max_depths_to_test:
                config    = GeneratorConfig(
                    beam_width=beam_width,
                    max_depth=max_depth,
                    top_k_results=instance.get("top_k", 10),
                )
                generator = TrajectoryGenerator(career_graph, config)

                logger.info(
                    f"[{instance['id']}] source={source} | "
                    f"beam={beam_width} | depth={max_depth} | "
                    f"profile={profile_name}"
                )

                start = time.perf_counter()
                try:
                    results    = generator.generate(source=source, constraints=constraint)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    metrics    = compute_metrics(
                        results=results,
                        config_name=f"beam{beam_width}_depth{max_depth}",
                        source_node=source,
                        constraint_profile=profile_name,
                        execution_time_ms=elapsed_ms,
                    )
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    logger.error(f"Error en {instance['id']}: {e}")
                    results    = []
                    metrics    = compute_metrics([], f"beam{beam_width}_depth{max_depth}",
                                                source, profile_name, elapsed_ms)

                # FIX [RE1]: LLM con timeout — no bloquea el experimento completo
                llm_result = None
                if llm_available and results and llm_analyzer:
                    try:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                llm_analyzer.rank_by,
                                results,
                                "mayor potencial de crecimiento profesional a largo plazo",
                                min(5, len(results)),
                            )
                            llm_result = future.result(timeout=LLM_TIMEOUT_SECS)
                        logger.debug(
                            f"LLM análisis completado ({llm_result.provider_used}): "
                            f"{llm_result.content[:100]}..."
                        )
                    except FuturesTimeoutError:
                        logger.warning(
                            f"LLM timeout ({LLM_TIMEOUT_SECS}s) en {instance['id']}. "
                            "Continuando sin análisis LLM para esta instancia."
                        )
                    except Exception as e:
                        logger.warning(f"Error LLM en {instance['id']}: {e}")

                exp_result = _extract_result(
                    instance_id=instance["id"],
                    source=source,
                    profile_name=profile_name,
                    beam_width=beam_width,
                    max_depth=max_depth,
                    use_llm=llm_result is not None,
                    results=results,
                    elapsed_ms=elapsed_ms,
                    metrics=metrics,
                    llm_result=llm_result,
                )
                all_results.append(exp_result)

    # Guardar CSV (FIX [R4]: en RESULTS_DIR unificado)
    if not all_results:
        logger.warning("No hay resultados para guardar.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV para análisis tabular
    csv_path = results_path
    fieldnames = list(asdict(all_results[0]).keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in all_results:
            writer.writerow(asdict(res))

    # JSON para visualizer.py (FIX [RE3]: formato compatible con runner.py)
    json_path = RESULTS_DIR / "run_experiments_results.json"
    json_data = [asdict(r) for r in all_results]
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))

    logger.success(
        f"Completado. {len(all_results)} registros guardados en:\n"
        f"  CSV:  {csv_path}\n"
        f"  JSON: {json_path}"
    )


if __name__ == "__main__":
    BASE_DIR       = Path(__file__).resolve().parent.parent.parent
    INSTANCES_FILE = BASE_DIR / "backend" / "data" / "test_instances.json"
    CSV_FILE       = RESULTS_DIR / "results.csv"

    if not INSTANCES_FILE.exists():
        logger.error(f"Archivo de instancias no encontrado: {INSTANCES_FILE}")
        logger.info("Crea primero backend/data/test_instances.json")
    else:
        run_experiments(INSTANCES_FILE, CSV_FILE)
