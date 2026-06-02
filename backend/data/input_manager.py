"""
backend/data/input_manager.py
-----------------------------
Gestor de inputs de usuario (carrera profesional + decisiones) con BD SQLite.

Permite:
  - Guardar/cargar configuraciones de usuario predefinidas
  - Análisis previo de inputs antes de exploración
  - Sugerencias de IA en tiempo real
  - Presets para pruebas rápidas
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


# ──────────────────────────────────────────────────────────────
# Schema de entrada
# ──────────────────────────────────────────────────────────────

@dataclass
class UserInput:
    """Entrada de usuario: carrera inicial + preferencias."""

    id: str
    source_career: str
    profile: str = "balanced"
    max_years: int = 12
    max_risk: float = 0.6
    beam_width: int = 10
    max_depth: int = 6
    top_k: int = 15
    use_simulation: bool = True
    user_profile_description: str = "profesional de tecnología"
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────
# Gestor de BD
# ──────────────────────────────────────────────────────────────

class InputManager:
    """Gestiona el almacenamiento y recuperación de inputs de usuario."""

    def __init__(self, db_path: str = "pathforge_inputs.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Inicializa la BD si no existe."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS user_inputs (
                id TEXT PRIMARY KEY,
                source_career TEXT NOT NULL,
                profile TEXT DEFAULT 'balanced',
                max_years INTEGER DEFAULT 12,
                max_risk REAL DEFAULT 0.6,
                beam_width INTEGER DEFAULT 10,
                max_depth INTEGER DEFAULT 6,
                top_k INTEGER DEFAULT 15,
                use_simulation INTEGER DEFAULT 1,
                user_profile_description TEXT DEFAULT 'profesional de tecnología',
                notes TEXT DEFAULT '',
                created_at TEXT,
                updated_at TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS ai_suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_id TEXT NOT NULL,
                suggestion TEXT NOT NULL,
                type TEXT DEFAULT 'recommendation',
                created_at TEXT,
                FOREIGN KEY (input_id) REFERENCES user_inputs(id)
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"BD inicializada en {self.db_path}")

    def save_input(self, user_input: UserInput) -> None:
        """Guarda o actualiza un input de usuario."""
        now = datetime.now().isoformat()
        if not user_input.created_at:
            user_input.created_at = now
        user_input.updated_at = now

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT OR REPLACE INTO user_inputs
            (id, source_career, profile, max_years, max_risk, beam_width, 
             max_depth, top_k, use_simulation, user_profile_description, notes,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_input.id,
            user_input.source_career,
            user_input.profile,
            user_input.max_years,
            user_input.max_risk,
            user_input.beam_width,
            user_input.max_depth,
            user_input.top_k,
            int(user_input.use_simulation),
            user_input.user_profile_description,
            user_input.notes,
            user_input.created_at,
            user_input.updated_at,
        ))

        conn.commit()
        conn.close()
        logger.success(f"Input '{user_input.id}' guardado")

    def load_input(self, input_id: str) -> Optional[UserInput]:
        """Carga un input por ID."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM user_inputs WHERE id = ?", (input_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return None

        return UserInput(
            id=row[0],
            source_career=row[1],
            profile=row[2],
            max_years=row[3],
            max_risk=row[4],
            beam_width=row[5],
            max_depth=row[6],
            top_k=row[7],
            use_simulation=bool(row[8]),
            user_profile_description=row[9],
            notes=row[10],
            created_at=row[11],
            updated_at=row[12],
        )

    def list_inputs(self) -> list[UserInput]:
        """Lista todos los inputs guardados."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM user_inputs ORDER BY updated_at DESC")
        rows = c.fetchall()
        conn.close()

        return [
            UserInput(
                id=row[0],
                source_career=row[1],
                profile=row[2],
                max_years=row[3],
                max_risk=row[4],
                beam_width=row[5],
                max_depth=row[6],
                top_k=row[7],
                use_simulation=bool(row[8]),
                user_profile_description=row[9],
                notes=row[10],
                created_at=row[11],
                updated_at=row[12],
            )
            for row in rows
        ]

    def delete_input(self, input_id: str) -> None:
        """Elimina un input y sus sugerencias."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM ai_suggestions WHERE input_id = ?", (input_id,))
        c.execute("DELETE FROM user_inputs WHERE id = ?", (input_id,))
        conn.commit()
        conn.close()
        logger.success(f"Input '{input_id}' eliminado")

    def save_suggestion(self, input_id: str, suggestion: str, stype: str = "recommendation") -> None:
        """Guarda una sugerencia de IA para un input."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now = datetime.now().isoformat()
        c.execute(
            "INSERT INTO ai_suggestions (input_id, suggestion, type, created_at) VALUES (?, ?, ?, ?)",
            (input_id, suggestion, stype, now)
        )
        conn.commit()
        conn.close()

    def get_suggestions(self, input_id: str) -> list[dict]:
        """Obtiene todas las sugerencias para un input."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "SELECT suggestion, type, created_at FROM ai_suggestions WHERE input_id = ? ORDER BY created_at DESC",
            (input_id,)
        )
        rows = c.fetchall()
        conn.close()
        return [{"text": r[0], "type": r[1], "timestamp": r[2]} for r in rows]

    def export_to_json(self, output_path: str) -> None:
        """Exporta todos los inputs a un archivo JSON."""
        inputs = self.list_inputs()
        data = {
            "exported_at": datetime.now().isoformat(),
            "inputs": [inp.to_dict() for inp in inputs],
        }
        Path(output_path).write_text(json.dumps(data, indent=2))
        logger.success(f"Inputs exportados a {output_path}")

    def import_from_json(self, input_path: str) -> None:
        """Importa inputs desde un archivo JSON."""
        data = json.loads(Path(input_path).read_text())
        for inp_dict in data.get("inputs", []):
            user_input = UserInput(**inp_dict)
            self.save_input(user_input)
        logger.success(f"Importados {len(data.get('inputs', []))} inputs")


# ──────────────────────────────────────────────────────────────
# Presets por defecto
# ──────────────────────────────────────────────────────────────

PRESET_INPUTS = [
    UserInput(
        id="preset_junior_conservative",
        source_career="junior_dev",
        profile="conservative",
        max_years=8,
        max_risk=0.3,
        user_profile_description="Desarrollador junior que busca estabilidad",
        notes="Prueba: busca riesgo bajo y crecimiento ordenado"
    ),
    UserInput(
        id="preset_senior_ambitious",
        source_career="senior_dev",
        profile="ambitious",
        max_years=10,
        max_risk=0.8,
        beam_width=15,
        top_k=20,
        user_profile_description="Senior con aspiraciones de liderazgo",
        notes="Prueba: riesgo alto para salario máximo"
    ),
    UserInput(
        id="preset_data_scientist",
        source_career="data_scientist",
        profile="balanced",
        max_years=12,
        max_risk=0.6,
        user_profile_description="Data Scientist buscando equilibrio",
        notes="Prueba: perfile equilibrado con IA"
    ),
]


def create_default_presets(manager: InputManager) -> None:
    """Crea los presets por defecto en la BD."""
    for preset in PRESET_INPUTS:
        if not manager.load_input(preset.id):
            manager.save_input(preset)
            logger.info(f"Preset '{preset.id}' creado")
