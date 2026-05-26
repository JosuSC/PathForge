"""test_quick.py — Verificación del core + LLM."""

from data.loader import load_career_graph
from core.graph import CareerGraph
from core.constraints import ConstraintProfiles
from core.generator import TrajectoryGenerator, GeneratorConfig
from core.evaluator import TrajectoryEvaluator
from llm.analyzer import TrajectoryAnalyzer

# 1. Generar trayectorias
G = load_career_graph()
career_graph = CareerGraph(G)
config = GeneratorConfig(beam_width=8, max_depth=5, top_k_results=5)
generator = TrajectoryGenerator(career_graph, config)

results = generator.generate(
    source="junior_dev",
    constraints=ConstraintProfiles.balanced(max_years=10),
)

print(f"Trayectorias generadas: {len(results)}")
for et in results:
    print(f"  {et.trajectory} | rank={et.pareto_rank}")

# 2. Análisis con Gemini
analyzer = TrajectoryAnalyzer(user_profile="desarrollador junior con 1 año de experiencia")

print("\n" + "="*60)
print("ANÁLISIS COMPARATIVO (Gemini)")
print("="*60)
result = analyzer.compare(results)
print(result.content)

print("\n" + "="*60)
print("RANKING POR CRITERIO")
print("="*60)
ranking = analyzer.rank_by(results, "quiero crecer rápido con el menor riesgo posible")
print(ranking.content)