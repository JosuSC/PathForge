# generate_problems.py
import json
from pathlib import Path
from collections import defaultdict, Counter
import re
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = Path("backend/data/problems")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_id(name: str) -> str:
    return re.sub(r'[^a-z0-9_]+', '_', name.lower().strip())

def infer_sector(job_title: str) -> str:
    t = job_title.lower()
    if any(w in t for w in ['health', 'medical', 'nurse', 'doctor', 'clinical', 'hospital', 'pharma']):
        return "healthcare"
    if any(w in t for w in ['finance', 'account', 'bank', 'investment', 'economic', 'audit']):
        return "finance"
    if any(w in t for w in ['software', 'engineer', 'developer', 'data', 'it', 'tech', 'programmer', 'devops']):
        return "technology"
    if any(w in t for w in ['education', 'teacher', 'professor', 'school', 'university', 'lecturer']):
        return "education"
    if any(w in t for w in ['legal', 'law', 'attorney', 'paralegal', 'solicitor']):
        return "legal"
    return "general"

def main():
    print("Cargando dataset 'ElenaSenger/Karrierewege_plus'...")
    ds = load_dataset("ElenaSenger/Karrierewege_plus", split="train")
    print(f"Dataset cargado: {len(ds)} registros.")

    # Ver columnas
    print("Columnas:", ds.column_names)
    # Esperamos: '_id', 'new_job_title_en_cp', 'experience_order', etc.

    # Agrupar por persona (_id) y ordenar por experience_order
    print("Agrupando trayectorias por persona...")
    persona_trayectorias = defaultdict(list)
    for record in tqdm(ds, desc="Procesando registros"):
        pid = record.get('_id')
        title = record.get('new_job_title_en_cp', '')
        order = record.get('experience_order', 0)
        if pid and title:
            persona_trayectorias[pid].append((order, title))

    # Ordenar cada lista por order y extraer solo títulos
    secuencias = []
    for pid, items in persona_trayectorias.items():
        items.sort(key=lambda x: x[0])  # orden por experience_order
        titles = [title for _, title in items]
        if len(titles) >= 2:
            secuencias.append(titles)

    print(f"Se construyeron {len(secuencias)} trayectorias con al menos 2 pasos.")

    # Contar transiciones y nodos por sector inferido
    sector_data = defaultdict(lambda: {"nodes": set(), "edges": Counter()})

    for seq in tqdm(secuencias, desc="Procesando trayectorias"):
        if not seq:
            continue
        # Inferir sector del primer trabajo (o podríamos usar el más frecuente)
        sector = infer_sector(seq[0])
        for i in range(len(seq)-1):
            from_job = seq[i]
            to_job = seq[i+1]
            sector_data[sector]["nodes"].add(from_job)
            sector_data[sector]["nodes"].add(to_job)
            sector_data[sector]["edges"][(from_job, to_job)] += 1

    print(f"Se encontraron {len(sector_data)} sectores.")
    saved = 0
    for sector, data in sector_data.items():
        if len(data["nodes"]) < 3:
            print(f"Sector '{sector}' con solo {len(data['nodes'])} nodos, omitido.")
            continue

        nodes = []
        for role in data["nodes"]:
            nodes.append({
                "id": sanitize_id(role),
                "label": role,
                "avg_salary": 0,
                "years_experience": 0,
                "skills": []
            })

        edges = []
        max_count = max(data["edges"].values()) if data["edges"] else 1
        for (from_role, to_role), count in data["edges"].items():
            risk = round(0.2 + 0.6 * (1 - count / max_count), 2)
            edges.append({
                "from": sanitize_id(from_role),
                "to": sanitize_id(to_role),
                "transition_years": 1,
                "difficulty": risk,
                "risk": risk
            })

        graph = {"nodes": nodes, "edges": edges}
        safe_sector = sanitize_id(sector)
        out_file = OUTPUT_DIR / f"{safe_sector}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        print(f"✅ Guardado: {out_file} (nodos={len(nodes)}, aristas={len(edges)})")
        saved += 1

    print(f"🎉 Proceso completado. {saved} problemas generados en {OUTPUT_DIR}")

if __name__ == "__main__":
    main()