# backend/experiments/run_all_domains.py
import sys
from pathlib import Path

# Asegurar que el backend esté en el path para poder importar tus módulos
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.runner import run_all_experiments

def main():
    # Ruta absoluta a backend/data/problems
    problems_dir = Path(__file__).resolve().parent.parent / "data" / "problems"
    
    if not problems_dir.exists():
        print(f"❌ Error: No se encontró el directorio {problems_dir}")
        return

    # Filtrar carpetas que tengan graph.json e instances.json
    domains = []
    for p in sorted(problems_dir.iterdir()):
        if p.is_dir() and (p / "graph.json").exists() and (p / "instances.json").exists():
            domains.append(p.name)
            
    print(f"🚀 === Encontrados {len(domains)} dominios para experimentar ===")
    
    # Iterar sobre cada dominio y lanzar los experimentos
    for i, domain_id in enumerate(domains, 1):
        print(f"\n{'='*20} [{i}/{len(domains)}] Dominio: {domain_id} {'='*20}")
        
        # Construir la ruta al archivo de instancias de este dominio
        instances_path = problems_dir / domain_id / "instances.json"
        
        # Ejecutar experimentos (usando la función que ya tienes)
        run_all_experiments(domain_id=domain_id, instances_path=instances_path)
        
    print("\n✅ === Todos los dominios han sido procesados ===")

if __name__ == "__main__":
    main()