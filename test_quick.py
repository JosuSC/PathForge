"""
test_quick.py
-------------
Verificación rápida del core + LLM (ejecutable directamente, sin pytest).
Ejecutar: python test_quick.py

FIX [TQ1]: importaciones absolutas con from backend.* + bootstrap de sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap de path — mismo patrón que el resto del proyecto
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data.loader import load_career_graph
from backend.core.graph import CareerGraph
from backend.core.constraints import ConstraintProfiles
from backend.core.generator import TrajectoryGenerator, GeneratorConfig
from backend.llm.analyzer import TrajectoryAnalyzer

# ── 1. Generar trayectorias ───────────────────────────────────

G            = load_career_graph()
career_graph = CareerGraph(G)
config       = GeneratorConfig(beam_width=8, max_depth=5, top_k_results=5)
generator    = TrajectoryGenerator(career_graph, config)

results = generator.generate(
    source="junior_dev",
    constraints=ConstraintProfiles.balanced(max_years=10),
)

print(f"\nTrayectorias generadas: {len(results)}")
for et in results:
    print(f"  {et.trajectory} | rank={et.pareto_rank} | "
          f"${et.scores.get('final_salary',0):,.0f} | "
          f"growth={et.scores.get('salary_growth',0):.1%}")

if not results:
    print("No se encontraron trayectorias. Revisa el grafo y las restricciones.")
    sys.exit(1)

# ── 2. Análisis con LLM ───────────────────────────────────────

try:
    analyzer = TrajectoryAnalyzer(
        user_profile="desarrollador junior con 1 año de experiencia"
    )

    print("\n" + "=" * 60)
    print("ANÁLISIS COMPARATIVO (LLM)")
    print("=" * 60)
    result = analyzer.compare(results)
    print(result.content)
    print(f"\n[Proveedor: {result.provider_used}]")

    print("\n" + "=" * 60)
    print("RANKING POR CRITERIO")
    print("=" * 60)
    ranking = analyzer.rank_by(results, "quiero crecer rápido con el menor riesgo posible")
    print(ranking.content)

except Exception as e:
    print(f"\n⚠ Análisis LLM no disponible: {e}")
    print("  Configura LLM_KEY_1=proveedor:api_key en .env para activarlo.")
