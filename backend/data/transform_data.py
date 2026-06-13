"""
transform_data.py
----------------------------
Genera ~20 dominios profesionales coherentes a partir del dataset Karrierewege_plus.
Agrupa ocupaciones por sectores amplios (salud, finanzas, hostelería, ingeniería, etc.)
y evita mezclas absurdas (taxidermistas en artes culinarias).

EJECUTAR:
    python backend/data/transform_data_v7_hybrid.py

OUTPUT:
    backend/data/problems/<dominio>/graph.json
    backend/data/problems/<dominio>/metadata.json
    backend/data/problems/<dominio>/instances.json
"""

from __future__ import annotations

import json
import csv
import re
import hashlib
import shutil
import sys
from pathlib import Path
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if (_SCRIPT_DIR / "raw").exists() or (_SCRIPT_DIR.name == "data"):
    DATA_DIR = _SCRIPT_DIR
else:
    DATA_DIR = _SCRIPT_DIR / "backend" / "data"

RAW_DIR = DATA_DIR / "raw"
PROBLEMS_DIR = DATA_DIR / "problems"
INPUT_CSV = RAW_DIR / "karrierewege_plus.csv"
TAXONOMY_JSON = RAW_DIR / "isco_esco_taxonomy.json"
SKILLS_JSON = RAW_DIR / "esco_skills.json"

MIN_NODES_PER_DOMAIN = 5       # dominios pequeños pero útiles
MIN_EDGES_PER_DOMAIN = 5       # mínimo para tener alguna trayectoria
MIN_TRANSITION_COUNT = 1

# ---------------------------------------------------------------------------
# DOMINIOS FINALES (20-25) – Agrupación amplia y coherente
# El mapeo ISCO-3 -> dominio se hará a través de ISCO-2 -> dominio
# y luego overrides adicionales.
# ---------------------------------------------------------------------------
FINAL_DOMAINS = {
    # Tecnología e informática
    "software_development",    # dev, data, cloud, cybersecurity, etc.
    "it_management",           # CIO, CTO, IT project managers
    "it_services",             # soporte, administración sistemas, redes
    "telecommunications",      # redes, telefonía, broadcasting

    # Ingeniería y construcción
    "engineering",             # ingenieros generales, civiles, mecánicos
    "engineering_technology",  # técnicos de ingeniería
    "electrical_engineering",  # electricidad, electrónica
    "architecture_design",     # arquitectos, diseñadores técnicos
    "construction",            # albañiles, instaladores, pintores

    # Salud y medicina
    "healthcare_professionals", # médicos, enfermeros, odontólogos
    "healthcare_technicians",   # técnicos de laboratorio, radiología
    "pharmacy",                 # farmacéuticos, ayudantes

    # Educación y ciencia
    "education",               # profesores, formadores
    "science",                 # biólogos, químicos, físicos
    "life_science_technology", # técnicos de laboratorio científico

    # Finanzas, administración y legal
    "finance",                 # contables, analistas financieros
    "administration",          # oficinistas, secretarios
    "legal_social",            # abogados, trabajadores sociales

    # Comercio, hostelería y servicios
    "retail_sales",            # vendedores, cajeros
    "hospitality",             # chefs, camareros, hotelería
    "logistics_transport",     # conductores, almacén, distribución

    # Arte, diseño y comunicación
    "arts_design",             # diseñadores, fotógrafos, artesanos
    "media_journalism",        # periodistas, productores

    # Sector primario y energía
    "agriculture",             # agricultores, pescadores
    "energy_mining",           # energías renovables, minería, petróleo

    # Seguridad y fuerzas armadas
    "protective_services",     # policía, bomberos
    "armed_forces",            # militares

    # Otros (solo si sobran datos)
    "personal_services",       # peluqueros, limpieza
    "waste_management",        # recogida de basuras
}

# ---------------------------------------------------------------------------
# Mapeo ISCO-2 (dos dígitos) -> dominio final (ampliado)
# Basado en la estructura ISCO-08, adaptado para coherencia.
# ---------------------------------------------------------------------------
ISCO2_TO_FINAL_DOMAIN = {
    # Fuerzas armadas
    "01": "armed_forces",
    "02": "armed_forces",

    # Directivos
    "11": "administration",        # legisladores, ejecutivos
    "12": "administration",        # gerentes admin
    "13": "engineering",           # gerentes producción/ingeniería
    "14": "hospitality",           # gerentes hostelería/retail

    # Profesionales científicos e ingenieros
    "21": "science",               # físicos, químicos, biólogos
    "22": "healthcare_professionals", # médicos, enfermeros
    "23": "education",             # profesores
    "24": "finance",               # economistas, administrativos
    "25": "software_development",  # ICT profesionales
    "26": "legal_social",          # abogados, periodistas, artistas

    # Técnicos
    "31": "engineering_technology", # técnicos ingeniería
    "32": "healthcare_technicians", # técnicos salud
    "33": "finance",               # técnicos financieros
    "34": "legal_social",          # técnicos legales/sociales
    "35": "it_services",           # técnicos ICT

    # Oficinistas
    "41": "administration",
    "42": "retail_sales",
    "43": "finance",
    "44": "administration",

    # Servicios y ventas
    "51": "hospitality",
    "52": "retail_sales",
    "53": "personal_services",
    "54": "protective_services",

    # Agricultura y pesca
    "61": "agriculture",
    "62": "agriculture",

    # Oficios construcción y metalurgia
    "71": "construction",
    "72": "engineering_technology",  # metalurgia, maquinaria
    "73": "arts_design",             # artesanía, impresión
    "74": "electrical_engineering",  # electricidad, electrónica
    "75": "manufacturing",           # procesamiento alimentos, madera

    # Operadores de planta y montadores
    "81": "energy_mining",
    "82": "manufacturing",
    "83": "logistics_transport",

    # Ocupaciones elementales
    "91": "personal_services",
    "92": "agriculture",
    "93": "construction",
    "94": "hospitality",
    "96": "waste_management",
}

# ---------------------------------------------------------------------------
# Overrides para casos específicos (incluye correcciones de contaminación)
# ---------------------------------------------------------------------------
OCCUPATION_OVERRIDES = {
    # Tecnología
    "artificial intelligence engineer": "software_development",
    "data scientist": "software_development",
    "data analyst": "software_development",
    "cloud architect": "software_development",
    "cybersecurity analyst": "software_development",
    "machine learning engineer": "software_development",

    # Energía renovable -> energy_mining
    "solar energy technician": "energy_mining",
    "wind turbine technician": "energy_mining",
    "renewable energy consultant": "energy_mining",

    # Prevención de contaminación: taxidermistas, fotógrafos, etc.
    "taxidermist": "arts_design",           # NO hospitality
    "photographic developer": "arts_design", # NO software_development
    "venue programmer": "arts_design",       # NO software_development
    "chef": "hospitality",
    "baker": "hospitality",
    "butcher": "hospitality",
    "cook": "hospitality",
    "waiter": "hospitality",
    "bartender": "hospitality",
    "museum curator": "arts_design",
    "archaeologist": "science",
    "veterinary": "healthcare_professionals",
    "animal caretaker": "agriculture",
}

# Palabras clave para excluir ocupaciones de ciertos dominios (evita mezclas)
DOMAIN_EXCLUSIONS = {
    "hospitality": ["taxiderm", "photograph", "curator", "archaeolog", "museum", "art", "sculptor"],
    "arts_design": ["chef", "baker", "butcher", "cook", "waiter", "bartender", "taxiderm", "veterinary"],
    "software_development": ["photographic", "venue programmer", "taxiderm"],
    "healthcare_professionals": ["taxiderm", "butcher"],
}

# Mapeo ISCO-3 a ISCO-2 (mismo que antes)
ISCO3_TO_ISCO2 = {
    "011": "01", "021": "02",
    "111": "11", "112": "11",
    "121": "12", "122": "12",
    "131": "13", "132": "13", "133": "13", "134": "13",
    "141": "14", "142": "14", "143": "14",
    "211": "21", "212": "21", "213": "21", "214": "21", "215": "21", "216": "21",
    "221": "22", "222": "22", "223": "22", "225": "22", "226": "22",
    "231": "23", "232": "23", "233": "23", "234": "23", "235": "23",
    "241": "24", "242": "24", "243": "24",
    "251": "25", "252": "25",
    "261": "26", "262": "26", "263": "26", "264": "26", "265": "26",
    "311": "31", "312": "31", "313": "31", "314": "31", "315": "31",
    "321": "32", "324": "32", "325": "32",
    "331": "33", "332": "33", "333": "33", "334": "33", "335": "33",
    "341": "34", "342": "34", "343": "34",
    "351": "35", "352": "35",
    "411": "41", "412": "41", "413": "41",
    "421": "42", "422": "42",
    "431": "43", "432": "43",
    "441": "44",
    "511": "51", "512": "51", "513": "51", "514": "51", "515": "51", "516": "51",
    "521": "52", "522": "52", "523": "52", "524": "52",
    "531": "53", "532": "53",
    "541": "54",
    "611": "61", "612": "61", "613": "61",
    "621": "62", "622": "62",
    "711": "71", "712": "71", "713": "71",
    "721": "72", "722": "72", "723": "72",
    "731": "73", "732": "73",
    "741": "74", "742": "74",
    "751": "75", "752": "75", "753": "75", "754": "75",
    "811": "81", "812": "81", "813": "81", "814": "81", "815": "81",
    "816": "81", "817": "81", "818": "81",
    "821": "82",
    "831": "83", "832": "83", "833": "83", "834": "83",
    "911": "91", "912": "91",
    "921": "92",
    "931": "93", "932": "93", "933": "93",
    "941": "94",
    "961": "96", "962": "96",
}

# Rangos salariales para cada dominio final (para estimaciones)
SALARY_RANGES = {
    "software_development":    (45000, 180000),
    "it_management":           (60000, 200000),
    "it_services":             (35000, 90000),
    "telecommunications":      (35000, 100000),
    "engineering":             (50000, 160000),
    "engineering_technology":  (40000, 100000),
    "electrical_engineering":  (45000, 120000),
    "architecture_design":     (40000, 130000),
    "construction":            (30000, 80000),
    "healthcare_professionals":(40000, 200000),
    "healthcare_technicians":  (30000, 70000),
    "pharmacy":                (35000, 110000),
    "education":               (28000, 90000),
    "science":                 (35000, 110000),
    "life_science_technology": (30000, 70000),
    "finance":                 (40000, 150000),
    "administration":          (25000, 65000),
    "legal_social":            (30000, 100000),
    "retail_sales":            (22000, 55000),
    "hospitality":             (22000, 70000),
    "logistics_transport":     (25000, 70000),
    "arts_design":             (25000, 90000),
    "media_journalism":        (28000, 80000),
    "agriculture":             (25000, 60000),
    "energy_mining":           (35000, 100000),
    "protective_services":     (30000, 80000),
    "armed_forces":            (25000, 70000),
    "personal_services":       (22000, 50000),
    "waste_management":        (25000, 60000),
    "other":                   (25000, 80000),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_taxonomy() -> dict:
    if not TAXONOMY_JSON.exists():
        print(f"  [ERROR] No se encontró {TAXONOMY_JSON}")
        return {}
    with open(TAXONOMY_JSON, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    primary_map = {}
    alt_map = {}
    for entry in taxonomy:
        esco_occ = entry.get("esco_occupation")
        if esco_occ is None:
            continue
        esco_lower = esco_occ.strip().lower()
        if esco_lower.startswith('"') and esco_lower.endswith('"'):
            esco_lower = esco_lower[1:-1]
        isco3 = str(entry.get("isco_code_3", "")).strip()
        isco2 = str(entry.get("isco_code_2", "")).strip()
        info = {"isco3": isco3, "isco2": isco2,
                "isco3_label": entry.get("isco_label_3", "") or "",
                "isco2_label": entry.get("isco_label_2", "") or ""}
        if esco_lower not in primary_map:
            primary_map[esco_lower] = info
        for alt_label in (entry.get("esco_labels") or []):
            alt_lower = alt_label.strip().lower()
            if alt_lower.startswith('"') and alt_lower.endswith('"'):
                alt_lower = alt_lower[1:-1]
            if alt_lower and alt_lower not in alt_map:
                alt_map[alt_lower] = info
    return {"primary": primary_map, "alt": alt_map}

def clean_esco_label(label: str) -> str:
    clean = label.strip().lower()
    if clean.startswith('"') and clean.endswith('"'):
        clean = clean[1:-1]
    return clean

def classify_occupation_final(esco_label: str, taxonomy: dict) -> str:
    """Clasifica una ocupación en uno de los dominios finales."""
    label_lower = clean_esco_label(esco_label)

    # 1. Overrides manuales
    if label_lower in OCCUPATION_OVERRIDES:
        return OCCUPATION_OVERRIDES[label_lower]

    # 2. Buscar ISCO-3 en taxonomy primario o alternativo
    isco3 = None
    primary = taxonomy.get("primary", {})
    alt = taxonomy.get("alt", {})
    if label_lower in primary:
        isco3 = primary[label_lower]["isco3"]
    elif label_lower in alt:
        isco3 = alt[label_lower]["isco3"]

    if isco3 and isco3 in ISCO3_TO_ISCO2:
        isco2 = ISCO3_TO_ISCO2[isco3]
        domain = ISCO2_TO_FINAL_DOMAIN.get(isco2, "other")
        # Verificar exclusiones por palabra clave
        if domain in DOMAIN_EXCLUSIONS:
            for bad in DOMAIN_EXCLUSIONS[domain]:
                if bad in label_lower:
                    # Reclasificar a otro dominio más apropiado
                    if domain == "hospitality" and any(a in label_lower for a in ["art", "photograph", "museum"]):
                        return "arts_design"
                    if domain == "arts_design" and any(a in label_lower for a in ["chef", "baker", "cook"]):
                        return "hospitality"
                    # Por defecto, lo enviamos a "other" para no contaminar
                    return "other"
        return domain

    return "other"

def get_broad_sector(domain: str) -> str:
    # Para metadata, derivamos un sector amplio (no crítico)
    mapping = {
        "software_development": "technology",
        "it_management": "technology",
        "it_services": "technology",
        "telecommunications": "technology",
        "engineering": "engineering",
        "engineering_technology": "engineering",
        "electrical_engineering": "engineering",
        "architecture_design": "engineering",
        "construction": "construction",
        "healthcare_professionals": "medical",
        "healthcare_technicians": "medical",
        "pharmacy": "medical",
        "education": "education",
        "science": "science",
        "life_science_technology": "science",
        "finance": "finance",
        "administration": "admin",
        "legal_social": "social",
        "retail_sales": "retail",
        "hospitality": "hospitality",
        "logistics_transport": "logistics",
        "arts_design": "arts",
        "media_journalism": "arts",
        "agriculture": "agriculture",
        "energy_mining": "energy",
        "protective_services": "security",
        "armed_forces": "security",
        "personal_services": "other",
        "waste_management": "other",
    }
    return mapping.get(domain, "other")

def make_node_id(esco_label: str) -> str:
    node_id = esco_label.lower().strip()
    node_id = re.sub(r'[^a-z0-9]+', '_', node_id)
    node_id = node_id.strip('_')
    if not node_id:
        node_id = hashlib.md5(esco_label.encode()).hexdigest()[:8]
    return node_id

def estimate_salary(esco_label: str, domain: str, position: float, sal_min: int, sal_max: int) -> int:
    base = sal_min + (sal_max - sal_min) * position
    noise = (hash(esco_label) % 100 - 50) * (sal_max - sal_min) * 0.002
    salary = int(base + noise)
    salary = round(salary / 1000) * 1000
    return max(salary, sal_min)

def load_esco_skills() -> dict:
    if not SKILLS_JSON.exists():
        return {}
    try:
        with open(SKILLS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def extract_skills(description: str, max_skills: int = 6) -> list[str]:
    if not description or len(description) < 20:
        return ["general"]
    desc = description.lower().strip()
    skill_patterns = [
        r'(?:proficient|skilled|experienced|knowledge|expertise)\s+(?:in|with|of)\s+([a-z\s,]+?)(?:\.|;|,|and)',
        r'(?:ability|capable)\s+to\s+([a-z\s]+?)(?:\.|;|,|and)',
        r'(?:using|use|operate|managing?|develop|design|implement|analyze|maintain|create|build|manage|lead|coordinate|supervise)\s+([a-z\s]+?)(?:\.|;|,|and)',
    ]
    skills = set()
    for pattern in skill_patterns:
        matches = re.findall(pattern, desc)
        for match in matches:
            for skill in re.split(r'[,;]|\band\b', match):
                skill = skill.strip()
                if 3 < len(skill) < 40 and not skill.startswith(('the ', 'a ', 'an ', 'to ')):
                    skills.add(skill)
    if not skills:
        keywords = re.findall(r'\b([a-z]{4,})\b', desc)
        stop_words = {'that', 'this', 'with', 'from', 'their', 'they', 'them',
                      'have', 'been', 'will', 'shall', 'must', 'should', 'could',
                      'would', 'other', 'which', 'where', 'when', 'what', 'each',
                      'work', 'role', 'also', 'such', 'into', 'more', 'over',
                      'these', 'those', 'while', 'some', 'than', 'may', 'its'}
        skills = {kw for kw in keywords[:20] if kw not in stop_words}
    result = sorted(skills)[:max_skills]
    return result if result else ["general"]

def create_problem(
    problem_id: str,
    domain: str,
    esco_labels: list[str],
    transitions: dict,
    esco_frequency: Counter,
    esco_descriptions: dict,
    esco_positions: dict,
    esco_skills_map: dict,
) -> bool:
    problem_dir = PROBLEMS_DIR / problem_id
    problem_dir.mkdir(parents=True, exist_ok=True)

    broad = get_broad_sector(domain)
    sal_min, sal_max = SALARY_RANGES.get(domain, (25000, 80000))
    max_freq = max(esco_frequency[e] for e in esco_labels) if esco_labels else 1

    # --- Nodos iniciales ---
    nodes = []
    node_id_set = set()
    for esco in esco_labels:
        node_id = make_node_id(esco)
        if node_id in node_id_set:
            node_id = f"{node_id}_{hash(esco) % 1000}"
        node_id_set.add(node_id)

        freq = esco_frequency.get(esco, 1)
        demand = round(freq / max_freq, 2)
        positions = esco_positions.get(esco, [0.5])
        avg_pos = sum(positions) / len(positions)
        salary = estimate_salary(esco, domain, avg_pos, sal_min, sal_max)
        satisfaction = round(0.5 + 0.3 * avg_pos + 0.1 * (salary / sal_max), 2)
        satisfaction = min(max(satisfaction, 0.3), 0.95)

        real_skills = esco_skills_map.get(esco, [])
        if real_skills and len(real_skills) >= 2:
            skills = real_skills[:6]
        else:
            descs = esco_descriptions.get(esco, [])
            best_desc = max(descs, key=len) if descs else ""
            skills = extract_skills(best_desc)

        years_exp = max(0, int(avg_pos * 15))
        if avg_pos < 0.25:
            node_type = "entry"
        elif avg_pos < 0.5:
            node_type = "mid"
        elif avg_pos < 0.75:
            node_type = "senior"
        else:
            node_type = "leadership"

        nodes.append({
            "id": node_id,
            "label": esco.title(),
            "type": node_type,
            "skills": skills,
            "avg_salary": salary,
            "demand": demand,
            "satisfaction": satisfaction,
            "years_experience": years_exp,
        })

    # --- Aristas iniciales ---
    esco_set = set(esco_labels)
    esco_to_node_id = {esco: node["id"] for esco, node in zip(esco_labels, nodes)}
    edges = []
    total_transitions_in_domain = sum(count for (f, t), count in transitions.items() if f in esco_set and t in esco_set)

    for (from_esco, to_esco), count in transitions.items():
        if from_esco not in esco_set or to_esco not in esco_set:
            continue
        from_id = esco_to_node_id.get(from_esco)
        to_id = esco_to_node_id.get(to_esco)
        if not from_id or not to_id or from_id == to_id:
            continue

        from_node = next((n for n in nodes if n["id"] == from_id), None)
        to_node = next((n for n in nodes if n["id"] == to_id), None)
        if not from_node or not to_node:
            continue

        salary_jump = (to_node["avg_salary"] - from_node["avg_salary"]) / max(from_node["avg_salary"], 1)
        salary_growth = round(max(0, salary_jump), 4)
        max_count = max(transitions.values()) if transitions else 1
        rarity_factor = max(0, 1 - count / max_count)
        difficulty = round(max(0.15, min(0.2 + abs(salary_jump) * 0.3 + rarity_factor * 0.3, 0.90)), 2)
        risk = round(max(0.10, min(0.15 + rarity_factor * 0.3 + abs(salary_jump) * 0.15, 0.85)), 2)
        transition_probability = count / max(total_transitions_in_domain, 1)
        exp_diff = abs(to_node["years_experience"] - from_node["years_experience"])
        trans_years = max(1, min(exp_diff + 1, 8))

        edges.append({
            "from": from_id,
            "to": to_id,
            "transition_probability": transition_probability,
            "transition_years": trans_years,
            "salary_growth": salary_growth,
            "difficulty": difficulty,
            "risk": risk,
        })

    # --- NUEVO: Eliminar nodos aislados (sin ninguna arista) ---
    nodes_with_edges = set()
    for e in edges:
        nodes_with_edges.add(e["from"])
        nodes_with_edges.add(e["to"])
    
    filtered_nodes = [n for n in nodes if n["id"] in nodes_with_edges]
    filtered_edges = edges  # las aristas ya solo referencian nodos con aristas, pero por si acaso
    
    # Reemplazar listas
    nodes = filtered_nodes
    edges = filtered_edges
    
    # Verificar mínimos después del filtrado
    if len(nodes) < MIN_NODES_PER_DOMAIN or len(edges) < MIN_EDGES_PER_DOMAIN:
        shutil.rmtree(problem_dir, ignore_errors=True)
        return False

    # --- Guardar graph.json ---
    with open(problem_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2, ensure_ascii=False)

    # --- Metadata actualizada ---
    metadata = {
        "id": problem_id,
        "domain": domain,
        "broad_sector": broad,
        "description": f"Career trajectories in {domain.replace('_', ' ')} derived from Karrierewege_plus dataset",
        "source": "ElenaSenger/Karrierewege_plus + ISCO-08 taxonomy",
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "salary_range": f"${sal_min:,} - ${sal_max:,}",
        "classification": "ISCO-08",
        "occupations": [n["label"] for n in nodes],
    }
    with open(problem_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return True

def generate_instances(problem_id: str, nodes: list[dict], edges: list[dict]) -> list[dict]:
    instances = []
    counter = 1
    nodes_with_successors = {e["from"] for e in edges}
    candidates = [n for n in nodes if n["type"] in ("entry", "mid") and n["id"] in nodes_with_successors]
    if not candidates:
        candidates = [n for n in nodes if n["id"] in nodes_with_successors]
    if not candidates:
        return []

    profiles = ["conservative", "ambitious", "balanced", "fast_track"]
    beam_widths = [3, 10]
    max_depths = [4, 8]

    for source in candidates[:3]:
        for profile in profiles:
            for bw in beam_widths:
                for md in max_depths:
                    max_years = 10 if profile == "balanced" else None
                    instances.append({
                        "id": f"{problem_id}_{counter:03d}",
                        "source_career": source["id"],
                        "target_career": None,
                        "profile": profile,
                        "max_years": max_years,
                        "beam_width": bw,
                        "max_depth": md,
                    })
                    counter += 1
    return instances

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PathForge - transform_data_v7_hybrid (dominios coherentes)")
    print("Genera ~20 dominios profesionales sin contaminación cruzada")
    print("=" * 70)

    if not INPUT_CSV.exists():
        print(f"\n[ERROR] No se encontró {INPUT_CSV}")
        sys.exit(1)

    # Limpiar problems
    print(f"\n[0/6] Limpiando directorio de problemas...")
    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)
    for item in PROBLEMS_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print("  Directorio limpio.")

    # Cargar taxonomy
    print(f"\n[1/6] Cargando taxonomy ISCO-ESCO...")
    taxonomy = load_taxonomy()
    if not taxonomy:
        print("[ERROR] No se pudo cargar el taxonomy.")
        sys.exit(1)
    print(f"  {len(taxonomy.get('primary', {}))} mapeos primarios, {len(taxonomy.get('alt', {}))} alternativos")

    # Skills opcionales
    print(f"\n[2/6] Cargando skills ESCO...")
    esco_skills_map = load_esco_skills()
    print(f"  {'Skills cargados: ' + str(len(esco_skills_map)) if esco_skills_map else 'Usando extracción por regex.'}")

    # Leer CSV
    print(f"\n[3/6] Leyendo {INPUT_CSV.name}...")
    trajectories = defaultdict(list)
    esco_frequency = Counter()
    esco_descriptions = defaultdict(list)
    all_transitions = Counter()
    row_count = 0
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            pid = row["_id"]
            esco = row.get("preferredLabel_en", "").strip()
            if not esco:
                continue
            esco_clean = clean_esco_label(esco)
            order = int(row.get("experience_order", 0))
            desc = row.get("description_en", "").strip()
            trajectories[pid].append((order, esco_clean))
            esco_frequency[esco_clean] += 1
            if desc and desc not in esco_descriptions[esco_clean]:
                esco_descriptions[esco_clean].append(desc)
    print(f"  {row_count:,} filas | {len(trajectories):,} personas | {len(esco_frequency):,} ocupaciones únicas")

    # Calcular transiciones y posiciones
    print(f"\n[4/6] Calculando transiciones...")
    esco_positions = defaultdict(list)
    for pid, steps in trajectories.items():
        steps_sorted = sorted(steps, key=lambda x: x[0])
        max_order = max(s[0] for s in steps_sorted)
        for i, (order, esco) in enumerate(steps_sorted):
            pos = order / max(max_order, 1)
            esco_positions[esco].append(pos)
            if i + 1 < len(steps_sorted):
                next_esco = steps_sorted[i+1][1]
                all_transitions[(esco, next_esco)] += 1
    transitions = {k: v for k, v in all_transitions.items() if v >= MIN_TRANSITION_COUNT}
    print(f"  {len(transitions):,} transiciones válidas")

    # Clasificar a dominio final
    print(f"\n[5/6] Clasificando ocupaciones a dominios coherentes...")
    esco_to_domain = {}
    unmatched = []
    for esco in esco_frequency:
        domain = classify_occupation_final(esco, taxonomy)
        esco_to_domain[esco] = domain
        if domain == "other":
            unmatched.append(esco)
    print(f"  {len(Counter(esco_to_domain.values()))} dominios detectados")
    print(f"  {len(unmatched)} ocupaciones en 'other' (descartadas)")

    # Agrupar ocupaciones por dominio
    domain_occupations = defaultdict(list)
    for esco, domain in esco_to_domain.items():
        if domain != "other":
            domain_occupations[domain].append(esco)

    # Crear problemas solo para los dominios que están en FINAL_DOMAINS (opcional, ya están filtrados)
    print(f"\n[6/6] Creando problemas (min {MIN_NODES_PER_DOMAIN} nodos, {MIN_EDGES_PER_DOMAIN} aristas)...")
    problems_created = 0
    all_instances = []
    sorted_domains = sorted(domain_occupations.items(), key=lambda x: -len(x[1]))

    for domain, occupations in sorted_domains:
        if domain not in FINAL_DOMAINS:
            continue  # solo crear los que definimos
        if len(occupations) < MIN_NODES_PER_DOMAIN:
            continue

        occ_set = set(occupations)
        domain_trans = {(f, t): c for (f, t), c in transitions.items() if f in occ_set and t in occ_set}

        success = create_problem(
            problem_id=domain,
            domain=domain,
            esco_labels=occupations,
            transitions=domain_trans,
            esco_frequency=esco_frequency,
            esco_descriptions=dict(esco_descriptions),
            esco_positions=dict(esco_positions),
            esco_skills_map=esco_skills_map,
        )
        if success:
            problems_created += 1
            with open(PROBLEMS_DIR / domain / "graph.json", encoding="utf-8") as f:
                graph = json.load(f)
            instances = generate_instances(domain, graph["nodes"], graph["edges"])
            with open(PROBLEMS_DIR / domain / "instances.json", "w", encoding="utf-8") as f:
                json.dump(instances, f, indent=2, ensure_ascii=False)
            all_instances.extend(instances)
            print(f"  [{problems_created:2d}] {domain:30s} | {len(graph['nodes']):3d} nodos, {len(graph['edges']):4d} aristas")
        else:
            print(f"  [SKIP] {domain:30s} (no cumple mínimos)")

    print(f"\n{'=' * 70}")
    print(f"COMPLETADO")
    print(f"  Dominios creados:     {problems_created}")
    print(f"  Instancias totales:   {len(all_instances)}")
    print(f"  Ocupaciones 'other':  {len(unmatched)} (descartadas)")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()