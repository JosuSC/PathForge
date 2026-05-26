"""test_quick.py — Verificación rápida del core (borrar después)."""

from data.loader import load_career_graph
from core.graph import CareerGraph
from core.constraints import ConstraintProfiles
from core.generator import TrajectoryGenerator, GeneratorConfig

# 1. Cargar grafo
G = load_career_graph()
career_graph = CareerGraph(G)
print(career_graph)

# 2. Generar trayectorias
config = GeneratorConfig(beam_width=8, max_depth=5, top_k_results=5)
generator = TrajectoryGenerator(career_graph, config)

results = generator.generate(
    source="junior_dev",
    constraints=ConstraintProfiles.balanced(max_years=10),
)

# 3. Mostrar resultados
for i, et in enumerate(results, 1):
    print(f"\n{'='*50}")
    print(f"#{i} Rank Pareto: {et.pareto_rank} | Crowding: {et.crowding_distance:.3f}")
    print(f"    Trayectoria: {et.trajectory}")
    print(f"    Scores: {et.scores}")
