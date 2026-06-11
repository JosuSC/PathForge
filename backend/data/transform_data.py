"""
transform_data_v6_FIXED.py
--------------------------
Corrige los problemas del v5:

1. Elimina dominios con 0 aristas (no se crean en disco)
2. Fija contaminacion cruzada en sector-names de industria:
   - motion_picture_production: elimina "production" generico
   - petroleum_coal_products: elimina renovables (van a renewable_energy)
   - insurance: elimina policy officers (van a government_leadership)
   - software_development: elimina photographic_developer, venue_programmer
   - medical_labs: elimina leather_laboratory_technician
3. Agrega transition_probability y salary_growth a aristas (estaban en v5 pero faltaban)
4. metadata.json usa 'domain' en vez de 'sector' (coherente con loader.py)
5. MIN_EDGES_PER_DOMAIN = 2 (minimo util para tener opciones de trayectoria)
6. Mejora instances.json: genera instancias validas desde nodos con sucesores
7. Elimina other0 - las ocupaciones sin clasificar se descartan en vez de mezclarlas
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

# FIX [05]: Rutas robustas independientes de dónde se ejecute el script.
# Si el script está en backend/data/ -> DATA_DIR es su propio directorio.
# Si está en la raíz del proyecto  -> DATA_DIR apunta a backend/data/.
_SCRIPT_DIR = Path(__file__).resolve().parent
if (_SCRIPT_DIR / "raw").exists() or (_SCRIPT_DIR.name == "data"):
    # Script está dentro de backend/data/
    DATA_DIR = _SCRIPT_DIR
else:
    # Script está en la raíz del proyecto (PathForge/)
    DATA_DIR = _SCRIPT_DIR / "backend" / "data"

RAW_DIR = DATA_DIR / "raw"
PROBLEMS_DIR = DATA_DIR / "problems"
INPUT_CSV = RAW_DIR / "karrierewege_plus.csv"
TAXONOMY_JSON = RAW_DIR / "isco_esco_taxonomy.json"
SKILLS_JSON = RAW_DIR / "esco_skills.json"

MIN_NODES_PER_DOMAIN = 3    # FIX: minimo 3 para tener trayectorias reales
MIN_EDGES_PER_DOMAIN = 2    # FIX: minimo 2 aristas (antes 1 era insuficiente)
MIN_TRANSITION_COUNT = 1

# ---------------------------------------------------------------------------
# Mapeo ISCO-3 -> Nombre de dominio (igual que v5)
# ---------------------------------------------------------------------------

ISCO3_DOMAIN_MAP = {
    "011": "armed_forces_officers", "021": "armed_forces_nco",
    "111": "government_leadership", "112": "executive_leadership",
    "121": "business_administration", "122": "sales_marketing_management",
    "131": "agriculture_management", "132": "industrial_management",
    "133": "it_management", "134": "professional_services_management",
    "141": "hospitality_management", "142": "retail_management",
    "143": "services_management",
    "211": "physical_sciences", "212": "mathematics_statistics",
    "213": "life_sciences", "214": "engineering",
    "215": "electrical_engineering", "216": "architecture_design",
    "221": "medical_doctors", "222": "nursing_midwifery",
    "223": "complementary_medicine", "225": "veterinary_medicine",
    "226": "allied_health",
    "231": "higher_education", "232": "vocational_education",
    "233": "secondary_education", "234": "primary_education",
    "235": "specialized_education",
    "241": "finance_professionals", "242": "business_administration_professionals",
    "243": "marketing_pr",
    "251": "software_development", "252": "database_networking",
    "261": "legal", "262": "library_information_science",
    "263": "social_religious_services", "264": "journalism_writing",
    "265": "creative_arts",
    "311": "engineering_technology", "312": "industrial_supervision",
    "313": "process_control", "314": "life_science_technology",
    "315": "transport_operations",
    "321": "medical_technology", "324": "veterinary_technology",
    "325": "health_technology",
    "331": "financial_services", "332": "sales_brokerage",
    "333": "business_services", "334": "administrative_support",
    "335": "government_regulation",
    "341": "legal_social_services", "342": "sports_fitness",
    "343": "cultural_culinary_arts",
    "351": "it_services", "352": "telecommunications",
    "411": "office_administration", "412": "secretarial_services",
    "413": "data_entry", "421": "banking_services",
    "422": "customer_service", "431": "accounting_clerical",
    "432": "logistics_clerical", "441": "clerical_support",
    "511": "travel_tourism", "512": "culinary_arts",
    "513": "food_service", "514": "beauty_services",
    "515": "building_services", "516": "personal_services",
    "521": "street_sales", "522": "retail_sales",
    "523": "cashier_services", "524": "sales",
    "531": "childcare", "532": "health_care_support",
    "541": "protective_services",
    "611": "crop_production", "612": "animal_husbandry",
    "613": "mixed_agriculture", "621": "forestry",
    "622": "fishing_hunting",
    "711": "structural_construction", "712": "finishing_construction",
    "713": "painting_cleaning_trades",
    "721": "metal_working", "722": "toolmaking",
    "723": "machinery_maintenance",
    "731": "handicrafts", "732": "printing",
    "741": "electrical_installation", "742": "electronics_installation",
    "751": "food_processing", "752": "woodworking",
    "753": "garment_making", "754": "crafts",
    "811": "mining_operations", "812": "metal_processing",
    "813": "chemical_processing", "814": "plastics_paper_processing",
    "815": "textile_processing", "816": "food_manufacturing",
    "817": "wood_paper_manufacturing", "818": "plant_operations",
    "821": "assembly",
    "831": "rail_transport", "832": "road_transport",
    "833": "heavy_vehicle_transport", "834": "mobile_plant_operations",
    "911": "cleaning_services", "912": "laundry_services",
    "921": "agricultural_labour", "931": "construction_labour",
    "932": "manufacturing_labour", "933": "transport_labour",
    "941": "food_preparation", "961": "waste_management",
    "962": "elementary_services",
}

# ---------------------------------------------------------------------------
# FIX: Overrides mejorados con mas precision
# ---------------------------------------------------------------------------

OCCUPATION_OVERRIDES = {
    # --- Inteligencia Artificial ---
    "artificial intelligence engineer": "artificial_intelligence",
    "data scientist": "artificial_intelligence",
    "data analyst": "artificial_intelligence",
    "data engineer": "artificial_intelligence",
    "bioinformatics scientist": "artificial_intelligence",
    "autonomous driving specialist": "artificial_intelligence",
    "machine learning engineer": "artificial_intelligence",

    # --- Energia Renovable ---
    "energy consultant": "renewable_energy",
    "energy manager": "renewable_energy",
    "offshore renewable energy technician": "renewable_energy",
    "onshore wind energy engineer": "renewable_energy",
    "solar energy technician": "renewable_energy",
    "solar energy sales consultant": "renewable_energy",
    "geothermal power plant operator": "renewable_energy",
    "renewable energy consultant": "renewable_energy",
    "domestic energy assessor": "renewable_energy",

    # --- Cloud/DevOps ---
    "cloud architect": "cloud_devops",
    "cloud devops engineer": "cloud_devops",

    # --- UX/UI Design ---
    "user interface designer": "ux_ui_design",
    "web content manager": "ux_ui_design",
    "web designer": "ux_ui_design",

    # --- Cybersecurity ---
    "chief ict security officer": "cybersecurity",
    "digital forensics expert": "cybersecurity",
    "cybersecurity risk manager": "cybersecurity",
    "data protection officer": "cybersecurity",

    # --- Database Administration ---
    "data warehouse designer": "database_administration",
    "database integrator": "database_administration",
    "data centre operator": "database_administration",

    # --- Robotics/Automation ---
    "robotics engineer": "robotics_automation",
    "robotics engineering technician": "robotics_automation",
    "automation engineer": "robotics_automation",
    "automation engineering technician": "robotics_automation",

    # --- Digital Media ---
    "digital media designer": "digital_media",
    "digital games designer": "digital_media",
    "podcast producer": "digital_media",

    # --- Networking ---
    "ict network administrator": "networking",
    "ict system administrator": "networking",
    "ict network technician": "networking",

    # --- Blockchain/Fintech ---
    "blockchain developer": "blockchain_fintech",

    # --- Tecnologia Sostenible ---
    "green ict consultant": "sustainable_technology",

    # --- IT Executive ---
    "chief information officer": "it_management",
    "chief technology officer": "it_management",
    "digital transformation manager": "it_management",

    # --- FIX: Evitar contaminacion en software_development ---
    # "photographic developer" NO es software -> va a su dominio ISCO (creative_arts)
    # "venue programmer" NO es software -> va a su dominio ISCO (cultural_culinary_arts)
    # (simplemente NO los ponemos en overrides, dejamos que ISCO los clasifique bien)
}

HARDCODED_ISCO3 = {
    "doctors surgery assistant": "325",
    "aftersales service manager": "132",
    "artificial intelligence engineer": "251",
    "building information modelling consultant": "311",
    "clinical trial assistant": "321",
    "corporate banking adviser": "241",
    "cybersecurity risk manager": "252",
    "digital transformation manager": "133",
    "furniture carpets and lighting equipment distribution manager": "142",
    "heating and ventilation service installer": "712",
    "podcast producer": "352",
    "process officer": "334",
    "vehicle restoration technician": "723",
    "web designer": "216",
    # FIX: Clasificar ocupaciones contaminantes correctamente
    "photographic developer": "731",         # artesania/handicrafts, NO software
    "venue programmer": "265",               # creative_arts, NO software
    "leather laboratory technician": "321",  # health_technology, NO medical_labs
    "medical laboratory technology vocational teacher": "235", # specialized_education
}

BROAD_SECTOR_MAP = {
    "armed_forces_officers": "public_administration",
    "armed_forces_nco": "public_administration",
    "government_leadership": "public_administration",
    "executive_leadership": "executive",
    "business_administration": "admin",
    "sales_marketing_management": "admin",
    "agriculture_management": "agriculture",
    "industrial_management": "manufacturing",
    "it_management": "technology",
    "professional_services_management": "professional_services",
    "hospitality_management": "hospitality",
    "retail_management": "retail",
    "services_management": "admin",
    "physical_sciences": "science",
    "mathematics_statistics": "finance",
    "life_sciences": "science",
    "engineering": "engineering",
    "electrical_engineering": "engineering",
    "architecture_design": "engineering",
    "medical_doctors": "medical",
    "nursing_midwifery": "medical",
    "complementary_medicine": "medical",
    "veterinary_medicine": "medical",
    "allied_health": "medical",
    "higher_education": "education",
    "vocational_education": "education",
    "secondary_education": "education",
    "primary_education": "education",
    "specialized_education": "education",
    "finance_professionals": "finance",
    "business_administration_professionals": "admin",
    "marketing_pr": "admin",
    "software_development": "technology",
    "database_networking": "technology",
    "artificial_intelligence": "technology",
    "cloud_devops": "technology",
    "ux_ui_design": "technology",
    "cybersecurity": "technology",
    "database_administration": "technology",
    "robotics_automation": "technology",
    "digital_media": "technology",
    "networking": "technology",
    "blockchain_fintech": "technology",
    "sustainable_technology": "technology",
    "renewable_energy": "energy",
    "legal": "legal",
    "library_information_science": "education",
    "social_religious_services": "social",
    "journalism_writing": "arts",
    "creative_arts": "arts",
    "engineering_technology": "engineering",
    "industrial_supervision": "manufacturing",
    "process_control": "manufacturing",
    "life_science_technology": "science",
    "transport_operations": "logistics",
    "medical_technology": "medical",
    "veterinary_technology": "medical",
    "health_technology": "medical",
    "financial_services": "finance",
    "sales_brokerage": "retail",
    "business_services": "admin",
    "administrative_support": "admin",
    "government_regulation": "public_administration",
    "legal_social_services": "social",
    "sports_fitness": "other",
    "cultural_culinary_arts": "arts",
    "it_services": "technology",
    "telecommunications": "technology",
    "office_administration": "admin",
    "secretarial_services": "admin",
    "data_entry": "admin",
    "banking_services": "finance",
    "customer_service": "admin",
    "accounting_clerical": "finance",
    "logistics_clerical": "logistics",
    "clerical_support": "admin",
    "travel_tourism": "hospitality",
    "culinary_arts": "hospitality",
    "food_service": "hospitality",
    "beauty_services": "other",
    "building_services": "admin",
    "personal_services": "other",
    "street_sales": "retail",
    "retail_sales": "retail",
    "cashier_services": "retail",
    "sales": "retail",
    "childcare": "other",
    "health_care_support": "medical",
    "protective_services": "public_administration",
    "crop_production": "agriculture",
    "animal_husbandry": "agriculture",
    "mixed_agriculture": "agriculture",
    "forestry": "agriculture",
    "fishing_hunting": "agriculture",
    "structural_construction": "construction",
    "finishing_construction": "construction",
    "painting_cleaning_trades": "construction",
    "metal_working": "manufacturing",
    "toolmaking": "manufacturing",
    "machinery_maintenance": "manufacturing",
    "handicrafts": "arts",
    "printing": "manufacturing",
    "electrical_installation": "construction",
    "electronics_installation": "technology",
    "food_processing": "manufacturing",
    "woodworking": "manufacturing",
    "garment_making": "manufacturing",
    "crafts": "arts",
    "mining_operations": "energy",
    "metal_processing": "manufacturing",
    "chemical_processing": "manufacturing",
    "plastics_paper_processing": "manufacturing",
    "textile_processing": "manufacturing",
    "food_manufacturing": "manufacturing",
    "wood_paper_manufacturing": "manufacturing",
    "plant_operations": "manufacturing",
    "assembly": "manufacturing",
    "rail_transport": "logistics",
    "road_transport": "logistics",
    "heavy_vehicle_transport": "logistics",
    "mobile_plant_operations": "logistics",
    "cleaning_services": "other",
    "laundry_services": "other",
    "agricultural_labour": "agriculture",
    "construction_labour": "construction",
    "manufacturing_labour": "manufacturing",
    "transport_labour": "logistics",
    "food_preparation": "hospitality",
    "waste_management": "other",
    "elementary_services": "other",
    # ISCO-2 merged parents
    "armed_forces": "public_administration",
    "legislators_executives": "executive",
    "business_managers": "admin",
    "production_managers": "manufacturing",
    "hospitality_retail_managers": "hospitality",
    "science_engineering_professionals": "science",
    "health_professionals": "medical",
    "teaching_professionals": "education",
    "business_professionals": "admin",
    "ict_professionals": "technology",
    "social_cultural_professionals": "social",
    "science_engineering_technicians": "engineering",
    "health_technicians": "medical",
    "business_technicians": "admin",
    "social_cultural_technicians": "social",
    "ict_technicians": "technology",
    "general_clerks": "admin",
    "customer_clerks": "admin",
    "numerical_clerks": "finance",
    "other_clerks": "admin",
    "personal_services_sector": "other",
    "sales_workers": "retail",
    "personal_care": "other",
    "protective_services_sector": "public_administration",
    "skilled_agriculture": "agriculture",
    "skilled_forestry_fishing": "agriculture",
    "building_trades": "construction",
    "metal_machinery_trades": "manufacturing",
    "handicraft_printing": "manufacturing",
    "electrical_trades": "construction",
    "processing_trades": "manufacturing",
    "stationary_plant_operators": "manufacturing",
    "assemblers": "manufacturing",
    "drivers_mobile_operators": "logistics",
    "cleaners_helpers": "other",
    "agricultural_labourers": "agriculture",
    "mining_construction_labourers": "construction",
    "food_preparation_assistants": "hospitality",
    "refuse_elementary_workers": "other",
}

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

ISCO2_DOMAIN_MAP = {
    "01": "armed_forces", "02": "armed_forces",
    "11": "legislators_executives", "12": "business_managers",
    "13": "production_managers", "14": "hospitality_retail_managers",
    "21": "science_engineering_professionals", "22": "health_professionals",
    "23": "teaching_professionals", "24": "business_professionals",
    "25": "ict_professionals", "26": "social_cultural_professionals",
    "31": "science_engineering_technicians", "32": "health_technicians",
    "33": "business_technicians", "34": "social_cultural_technicians",
    "35": "ict_technicians",
    "41": "general_clerks", "42": "customer_clerks",
    "43": "numerical_clerks", "44": "other_clerks",
    "51": "personal_services_sector", "52": "sales_workers",
    "53": "personal_care", "54": "protective_services_sector",
    "61": "skilled_agriculture", "62": "skilled_forestry_fishing",
    "71": "building_trades", "72": "metal_machinery_trades",
    "73": "handicraft_printing", "74": "electrical_trades",
    "75": "processing_trades",
    "81": "stationary_plant_operators", "82": "assemblers",
    "83": "drivers_mobile_operators",
    "91": "cleaners_helpers", "92": "agricultural_labourers",
    "93": "mining_construction_labourers", "94": "food_preparation_assistants",
    "96": "refuse_elementary_workers",
}

SALARY_RANGES = {
    "technology":       (45000, 180000),
    "medical":          (35000, 350000),
    "finance":          (40000, 250000),
    "education":        (28000, 120000),
    "legal":            (35000, 200000),
    "engineering":      (50000, 160000),
    "manufacturing":    (28000,  75000),
    "construction":     (30000,  90000),
    "hospitality":      (22000,  70000),
    "logistics":        (25000,  70000),
    "retail":           (22000,  55000),
    "admin":            (25000,  65000),
    "agriculture":      (25000,  65000),
    "arts":             (25000, 100000),
    "science":          (35000, 130000),
    "security":         (25000,  60000),
    "other":            (25000,  80000),
    "energy":           (35000, 120000),
    "nonprofit":        (25000,  90000),
    "real_estate":      (35000, 150000),
    "professional_services": (40000, 130000),
    "public_administration": (30000, 110000),
    "executive":        (60000, 300000),
    "social":           (25000,  80000),
}

# ---------------------------------------------------------------------------
# Helpers (igual que v5)
# ---------------------------------------------------------------------------

def load_taxonomy() -> dict:
    if not TAXONOMY_JSON.exists():
        print(f"  [ERROR] No se encontro {TAXONOMY_JSON}")
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


def classify_occupation(esco_label: str, taxonomy: dict) -> str:
    label_lower = clean_esco_label(esco_label)
    if label_lower in OCCUPATION_OVERRIDES:
        return OCCUPATION_OVERRIDES[label_lower]
    if label_lower in HARDCODED_ISCO3:
        isco3 = HARDCODED_ISCO3[label_lower]
        return ISCO3_DOMAIN_MAP.get(isco3, "other")
    primary = taxonomy.get("primary", {})
    if label_lower in primary:
        isco3 = primary[label_lower]["isco3"]
        return ISCO3_DOMAIN_MAP.get(isco3, "other")
    alt = taxonomy.get("alt", {})
    if label_lower in alt:
        isco3 = alt[label_lower]["isco3"]
        return ISCO3_DOMAIN_MAP.get(isco3, "other")
    return "other"


def get_broad_sector(domain: str) -> str:
    return BROAD_SECTOR_MAP.get(domain, "other")


def make_node_id(esco_label: str) -> str:
    node_id = esco_label.lower().strip()
    node_id = re.sub(r'[^a-z0-9]+', '_', node_id)
    node_id = node_id.strip('_')
    if not node_id:
        node_id = hashlib.md5(esco_label.encode()).hexdigest()[:8]
    return node_id


def estimate_salary(esco_label: str, domain: str, position: float,
                    salary_min: int, salary_max: int) -> int:
    base = salary_min + (salary_max - salary_min) * position
    noise = (hash(esco_label) % 100 - 50) * (salary_max - salary_min) * 0.002
    salary = int(base + noise)
    salary = round(salary / 1000) * 1000
    return max(salary, salary_min)


def load_esco_skills() -> dict:
    if not SKILLS_JSON.exists():
        return {}
    try:
        with open(SKILLS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
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


# ---------------------------------------------------------------------------
# Crear problema (FIX: agrega transition_probability + salary_growth en aristas,
#                       usa 'domain' en metadata, descarta dominios con < MIN_EDGES)
# ---------------------------------------------------------------------------

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
    sal_min, sal_max = SALARY_RANGES.get(broad, (25000, 80000))
    max_freq = max(esco_frequency[e] for e in esco_labels) if esco_labels else 1

    # --- Nodos ---
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

    # --- Aristas (FIX: incluye transition_probability y salary_growth) ---
    esco_set = set(esco_labels)
    node_ids = {n["id"] for n in nodes}
    esco_to_node_id = {esco: node["id"] for esco, node in zip(esco_labels, nodes)}

    edges = []
    total_transitions_in_domain = sum(
        count for (from_e, to_e), count in transitions.items()
        if from_e in esco_set and to_e in esco_set
    )

    for (from_esco, to_esco), count in transitions.items():
        if from_esco not in esco_set or to_esco not in esco_set:
            continue
        from_id = esco_to_node_id.get(from_esco)
        to_id = esco_to_node_id.get(to_esco)
        if not from_id or not to_id:
            continue
        if from_id not in node_ids or to_id not in node_ids:
            continue
        if from_id == to_id:
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
        transition_probability = round(count / max(total_transitions_in_domain, 1), 4)
        exp_diff = abs(to_node["years_experience"] - from_node["years_experience"])
        trans_years = max(1, min(exp_diff + 1, 8))

        edges.append({
            "from": from_id,
            "to": to_id,
            "transition_probability": transition_probability,  # FIX: incluido
            "transition_years": trans_years,
            "salary_growth": salary_growth,                    # FIX: incluido
            "difficulty": difficulty,
            "risk": risk,
        })

    # FIX: Validar minimos mas estrictos
    if len(nodes) < MIN_NODES_PER_DOMAIN:
        shutil.rmtree(problem_dir, ignore_errors=True)
        return False
    if len(edges) < MIN_EDGES_PER_DOMAIN:
        # FIX: Dominios con 0 o 1 aristas se descartan (inutiles para trayectorias)
        shutil.rmtree(problem_dir, ignore_errors=True)
        return False

    # --- graph.json ---
    with open(problem_dir / "graph.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2, ensure_ascii=False)

    # --- metadata.json (FIX: usa 'domain' en vez de 'sector') ---
    metadata = {
        "id": problem_id,
        "domain": domain,              # FIX: clave 'domain' (coherente con loader.py)
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


# ---------------------------------------------------------------------------
# Instancias (FIX: genera mas instancias y asegura que source_career exista)
# ---------------------------------------------------------------------------

def generate_instances(problem_id: str, nodes: list[dict], edges: list[dict]) -> list[dict]:
    instances = []
    counter = 1

    nodes_with_successors = {e["from"] for e in edges}
    candidates = [n for n in nodes
                  if n["type"] in ("entry", "mid") and n["id"] in nodes_with_successors]
    if not candidates:
        candidates = [n for n in nodes if n["id"] in nodes_with_successors]
    if not candidates:
        return []  # FIX: si no hay nodos con sucesores, no generar instancias vacias

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
                        "source_career": source["id"],  # FIX: garantizado que existe en graph
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
    print("PathForge - transform_data_v6_FIXED")
    print("Fixes: sin 0-edge domains, anti-contaminacion, transition_probability")
    print("=" * 70)

    if not INPUT_CSV.exists():
        print(f"\n[ERROR] No se encontro {INPUT_CSV}")
        print("Extrae el data.rar en backend/data/raw/ y vuelve a ejecutar.")
        sys.exit(1)

    # Limpiar problems dir
    print(f"\n[0/6] Limpiando directorio de problemas...")
    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)
    for item in PROBLEMS_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file():
            item.unlink()
    print("  Directorio limpio.")

    # Taxonomy
    print(f"\n[1/6] Cargando taxonomy ISCO-ESCO...")
    taxonomy = load_taxonomy()
    if not taxonomy:
        print("[ERROR] No se pudo cargar el taxonomy.")
        sys.exit(1)
    print(f"  {len(taxonomy.get('primary', {}))} mapeos primarios, "
          f"{len(taxonomy.get('alt', {}))} alternativos")

    # ESCO Skills
    print(f"\n[2/6] Cargando skills ESCO...")
    esco_skills_map = load_esco_skills()
    print(f"  {'Skills cargados: ' + str(len(esco_skills_map)) if esco_skills_map else 'Usando extraccion por regex.'}")

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

    print(f"  {row_count:,} filas | {len(trajectories):,} personas | "
          f"{len(esco_frequency):,} ocupaciones unicas")

    # Transiciones y posiciones
    print(f"\n[4/6] Calculando transiciones...")
    esco_positions = defaultdict(list)

    for pid, steps in trajectories.items():
        steps_sorted = sorted(steps, key=lambda x: x[0])
        max_order = max(s[0] for s in steps_sorted)
        for i in range(len(steps_sorted)):
            _, esco = steps_sorted[i]
            pos = steps_sorted[i][0] / max(max_order, 1)
            esco_positions[esco].append(pos)
            if i + 1 < len(steps_sorted):
                _, next_esco = steps_sorted[i + 1]
                all_transitions[(esco, next_esco)] += 1

    transitions = {k: v for k, v in all_transitions.items()
                   if v >= MIN_TRANSITION_COUNT}
    print(f"  {len(transitions):,} transiciones validas")

    # Clasificar
    print(f"\n[5/6] Clasificando ocupaciones con ISCO-08...")
    esco_to_domain = {}
    unmatched = []

    for esco in esco_frequency:
        domain = classify_occupation(esco, taxonomy)
        esco_to_domain[esco] = domain
        if domain == "other":
            unmatched.append(esco)

    print(f"  {len(Counter(esco_to_domain.values()))} dominios detectados")
    print(f"  {len(unmatched)} ocupaciones en 'other' (descartadas)")

    # Merge dominios de 1-2 nodos con padre ISCO-2
    domain_occupations = defaultdict(list)
    for esco, domain in esco_to_domain.items():
        domain_occupations[domain].append(esco)

    print(f"\n  Fusionando dominios pequenos con padres ISCO-2...")
    merged_domains = {}
    for domain, occupations in list(domain_occupations.items()):
        if domain == "other":
            continue
        if len(occupations) < MIN_NODES_PER_DOMAIN:
            parent_isco2 = None
            for isco3, dom in ISCO3_DOMAIN_MAP.items():
                if dom == domain:
                    parent_isco2 = ISCO3_TO_ISCO2.get(isco3)
                    break
            if parent_isco2 is None:
                for occ in occupations:
                    ll = clean_esco_label(occ)
                    if ll in HARDCODED_ISCO3:
                        parent_isco2 = ISCO3_TO_ISCO2.get(HARDCODED_ISCO3[ll])
                        break
                    if ll in taxonomy.get("primary", {}):
                        parent_isco2 = ISCO3_TO_ISCO2.get(taxonomy["primary"][ll]["isco3"])
                        break
            if parent_isco2:
                parent_domain = ISCO2_DOMAIN_MAP.get(parent_isco2, "other")
                merged_domains[domain] = parent_domain

    for esco, domain in esco_to_domain.items():
        if domain in merged_domains:
            esco_to_domain[esco] = merged_domains[domain]

    domain_occupations = defaultdict(list)
    for esco, domain in esco_to_domain.items():
        domain_occupations[domain].append(esco)

    # Crear problemas
    print(f"\n[6/6] Creando problemas "
          f"(min {MIN_NODES_PER_DOMAIN} nodos, min {MIN_EDGES_PER_DOMAIN} aristas)...")
    problems_created = 0
    discarded_no_edges = 0
    all_instances = []

    sorted_domains = sorted(domain_occupations.items(), key=lambda x: -len(x[1]))

    for domain, occupations in sorted_domains:
        if domain == "other":
            continue
        if len(occupations) < MIN_NODES_PER_DOMAIN:
            continue

        occ_set = set(occupations)
        domain_trans = {(f, t): c for (f, t), c in transitions.items()
                        if f in occ_set and t in occ_set}

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
            with open(PROBLEMS_DIR / domain / "graph.json") as f:
                graph = json.load(f)
            instances = generate_instances(domain, graph["nodes"], graph["edges"])

            # FIX: guardar como lista directa (no dict con key 'instances')
            with open(PROBLEMS_DIR / domain / "instances.json", "w", encoding="utf-8") as f:
                json.dump(instances, f, indent=2, ensure_ascii=False)
            all_instances.extend(instances)

            n, e = len(graph["nodes"]), len(graph["edges"])
            broad = get_broad_sector(domain)
            print(f"  [{problems_created:3d}] {domain:45s} | {n:3d} nodos, {e:4d} aristas | {broad}")
        else:
            discarded_no_edges += 1

    print(f"\n{'=' * 70}")
    print(f"COMPLETADO")
    print(f"  Dominios creados:     {problems_created}")
    print(f"  Descartados (pocos nodos/aristas): {discarded_no_edges}")
    print(f"  Fusionados:           {len(merged_domains)}")
    print(f"  Instancias totales:   {len(all_instances)}")
    print(f"  Ocupaciones 'other':  {len(unmatched)} (descartadas, sin dominio)")
    print(f"{'=' * 70}")

    # Verificar dominios clave
    key_domains = ["artificial_intelligence", "cybersecurity", "cloud_devops",
                   "renewable_energy", "software_development", "engineering",
                   "medical_doctors", "finance_professionals", "legal"]
    print(f"\nVerificacion dominios clave:")
    for kd in key_domains:
        domain_dir = PROBLEMS_DIR / kd
        if domain_dir.exists():
            with open(domain_dir / "graph.json") as f:
                g = json.load(f)
            print(f"  {kd:30s}: {len(g['nodes']):3d} nodos, {len(g['edges']):4d} aristas ✓")
        else:
            print(f"  {kd:30s}: NO CREADO")


if __name__ == "__main__":
    main()