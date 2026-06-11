"""
download_data.py 
-------------------
Descarga datos crudos de carreras profesionales desde HuggingFace
y los guarda en backend/data/raw/.

QUE HACE:
1. Descarga el dataset "Karrierewege_plus" de HuggingFace
   - 80,000 trayectorias profesionales reales
   - 1,162 ocupaciones ESCO unicas
   - Promedio de 5 pasos por trayectoria
   - Datos de Flandes (Belgica), servicio publico de empleo VDAB
2. Lo guarda como CSV en backend/data/raw/karrierewege_plus.csv
3. Descarga el taxonomy ISCO-ESCO para clasificacion de ocupaciones
   - 3,017 mapeos ESCO -> ISCO-08
   - Clasificacion autoritativa (no keywords)
   - Guardado como JSON en raw/isco_esco_taxonomy.json
4. Opcionalmente extrae skills del dataset base Karrierewege
   - Guarda esco_skills.json con skills ESCO por ocupacion
5. Genera un resumen estadistico en raw/dataset_summary.json

EJECUCION:
    cd PathForge
    python backend/data/download_data.py

OUTPUT:
    backend/data/raw/karrierewege_plus.csv  (datos crudos - trayectorias)
    backend/data/raw/isco_esco_taxonomy.json (mapeo ESCO -> ISCO)
    backend/data/raw/esco_skills.json        (skills ESCO - opcional)
    backend/data/raw/dataset_summary.json    (resumen)
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent  # backend/data/
RAW_DIR = DATA_DIR / "raw"
DATASET_NAME = "ElenaSenger/Karrierewege_plus"
TAXONOMY_DATASET = "ICILS/isco_esco_occupations_taxonomy"
BASE_DATASET = "ElenaSenger/Karrierewege"

OUTPUT_CSV = RAW_DIR / "karrierewege_plus.csv"
TAXONOMY_JSON = RAW_DIR / "isco_esco_taxonomy.json"
SKILLS_JSON = RAW_DIR / "esco_skills.json"
SUMMARY_JSON = RAW_DIR / "dataset_summary.json"


# ---------------------------------------------------------------------------
# Funciones de descarga
# ---------------------------------------------------------------------------

def download_karrierewege_plus():
    """Descarga el dataset Karrierewege_plus (trayectorias + predicciones LLM)."""
    if OUTPUT_CSV.exists():
        size_mb = OUTPUT_CSV.stat().st_size / (1024 * 1024)
        print(f"\n  [SKIP] {OUTPUT_CSV} ya existe.")
        print(f"  Tamano: {size_mb:.1f} MB")
        print(f"  Para re-descargar, borra el archivo primero.")
        return True

    print(f"\n  Descargando dataset '{DATASET_NAME}' de HuggingFace...")
    print("  (Esto puede tardar 1-3 minutos la primera vez)")

    try:
        from datasets import load_dataset
    except ImportError:
        print("\n  [ERROR] Necesitas instalar el paquete 'datasets':")
        print("  pip install datasets")
        return False

    ds = load_dataset(DATASET_NAME, split="train")
    print(f"  Descargado: {len(ds)} filas")

    print(f"\n  Guardando CSV en: {OUTPUT_CSV}")
    columns = ds.column_names
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i, row in enumerate(ds):
            writer.writerow(row)
            if (i + 1) % 100000 == 0:
                print(f"    Progreso: {i + 1}/{len(ds)} filas")

    size_mb = OUTPUT_CSV.stat().st_size / (1024 * 1024)
    print(f"  CSV guardado: {size_mb:.1f} MB")
    return True


def download_isco_taxonomy():
    """Descarga el taxonomy ISCO-ESCO para clasificacion de ocupaciones."""
    if TAXONOMY_JSON.exists():
        print(f"\n  [SKIP] {TAXONOMY_JSON} ya existe.")
        print(f"  Para re-descargar, borra el archivo primero.")
        return True

    print(f"\n  Descargando taxonomy ISCO-ESCO de HuggingFace...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("\n  [ERROR] Necesitas instalar el paquete 'datasets':")
        print("  pip install datasets")
        return False

    ds = load_dataset(TAXONOMY_DATASET, "isco_taxonomy", split="train")
    print(f"  Descargado: {len(ds)} mapeos ESCO->ISCO")

    taxonomy = []
    for row in ds:
        entry = {
            "esco_occupation": row.get("ESCO_OCCUPATION"),
            "esco_description": row.get("ESCO_DESCRIPTION"),
            "esco_code": str(row.get("ESCO_CODE", "")),
            "esco_labels": row.get("ESCO_LABELS", []),
            "isco_code_1": str(row.get("ISCO_CODE_1", "")),
            "isco_code_2": str(row.get("ISCO_CODE_2", "")),
            "isco_code_3": str(row.get("ISCO_CODE_3", "")),
            "isco_code_4": str(row.get("ISCO_CODE_4", "")),
            "isco_label_1": row.get("ISCO_LABEL_1"),
            "isco_label_2": row.get("ISCO_LABEL_2"),
            "isco_label_3": row.get("ISCO_LABEL_3"),
            "isco_label_4": row.get("ISCO_LABEL_4"),
        }
        taxonomy.append(entry)

    with open(TAXONOMY_JSON, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    print(f"  Taxonomy guardado en: {TAXONOMY_JSON}")
    return True


def download_esco_skills():
    """Extrae skills ESCO del dataset base Karrierewege (opcional)."""
    if SKILLS_JSON.exists():
        print(f"\n  [SKIP] {SKILLS_JSON} ya existe.")
        print(f"  Para re-descargar, borra el archivo primero.")
        return True

    print(f"\n  Descargando skills ESCO del dataset base Karrierewege...")
    print("  (Esto puede tardar 5-10 minutos, dataset de ~2M filas)")
    print("  Este paso es OPCIONAL. Si falla, el transform usara regex para skills.")

    try:
        from datasets import load_dataset
    except ImportError:
        print("\n  [WARNING] No se pudo importar 'datasets'. Saltando skills.")
        return False

    try:
        ds = load_dataset(BASE_DATASET, split="train")
        print(f"  Descargado: {len(ds)} filas")
    except Exception as e:
        print(f"  [WARNING] Error descargando base dataset: {e}")
        print("  Saltando extraccion de skills. El transform usara regex.")
        return False

    skills_map = defaultdict(set)
    for i, row in enumerate(ds):
        esco = row.get("preferredLabel_en", "").strip()
        skills_raw = row.get("skills", "")
        if esco and skills_raw:
            try:
                parsed = json.loads(skills_raw) if isinstance(skills_raw, str) else skills_raw
                if isinstance(parsed, list):
                    for skill in parsed:
                        if isinstance(skill, str) and len(skill.strip()) > 2:
                            skills_map[esco].add(skill.strip())
            except (json.JSONDecodeError, TypeError):
                pass
        if (i + 1) % 500000 == 0:
            print(f"    Progreso: {i + 1}/{len(ds)} filas")

    # Convert sets to sorted lists
    skills_json = {k: sorted(v) for k, v in skills_map.items()}

    with open(SKILLS_JSON, "w", encoding="utf-8") as f:
        json.dump(skills_json, f, indent=2, ensure_ascii=False)

    print(f"  Skills extraidos para {len(skills_json)} ocupaciones ESCO")
    print(f"  Guardado en: {SKILLS_JSON}")
    return True


def generate_summary():
    """Genera un resumen estadistico del dataset."""
    if not OUTPUT_CSV.exists():
        print("\n  [SKIP] No se encontro el CSV para generar resumen.")
        return

    print(f"\n  Generando resumen estadistico...")

    trajectories = defaultdict(list)
    esco_counts = Counter()
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for row in reader:
            trajectories[row['_id']].append({
                'job_title': row.get('preferredLabel_en', ''),   # FIX: consistente con transform_data.py
                'esco_label': row.get('preferredLabel_en', ''),
                'experience_order': row.get('experience_order', ''),
            })
            esco = row.get('preferredLabel_en', '').strip()
            if esco:
                esco_counts[esco] += 1

    lengths = [len(v) for v in trajectories.values()]

    # Check taxonomy status
    taxonomy_status = "available" if TAXONOMY_JSON.exists() else "not downloaded"
    skills_status = "available" if SKILLS_JSON.exists() else "not downloaded"

    summary = {
        "dataset_name": DATASET_NAME,
        "total_rows": sum(esco_counts.values()),
        "unique_trajectories": len(trajectories),
        "unique_occupations": len(esco_counts),
        "trajectory_length_stats": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
        },
        "top_30_occupations": dict(esco_counts.most_common(30)),
        "taxonomy_status": taxonomy_status,
        "skills_status": skills_status,
        "columns": columns,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Resumen guardado en: {SUMMARY_JSON}")
    print(f"  Ocupaciones: {len(esco_counts)}")
    print(f"  Trayectorias: {len(trajectories)}")
    print(f"  Taxonomy ISCO: {taxonomy_status}")
    print(f"  Skills ESCO: {skills_status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PathForge - Descarga de Datos Crudos (v3)")
    print("=" * 60)

    # Crear directorio
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Paso 1: Descargar Karrierewege_plus
    print("\n[1/3] Descargando trayectorias profesionales...")
    download_karrierewege_plus()

    # Paso 2: Descargar taxonomy ISCO-ESCO
    print("\n[2/3] Descargando taxonomy ISCO-ESCO...")
    download_isco_taxonomy()

    # Paso 3: Extraer skills ESCO (opcional)
    print("\n[3/3] Extrayendo skills ESCO (opcional)...")
    download_esco_skills()

    # Generar resumen
    generate_summary()

    print("\n" + "=" * 60)
    print("Descarga completada!")
    print(f"  Archivos en {RAW_DIR}:")
    for f in sorted(RAW_DIR.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name} ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()