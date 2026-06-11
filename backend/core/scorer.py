"""
backend/core/scorer.py
----------------------
Modelo de Machine Learning para scoring de transiciones de carrera.

"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn no disponible. Usando scoring heurístico.")


# ──────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────

def transition_to_features(
    from_attrs: dict,
    to_attrs:   dict,
    edge_attrs: dict,
) -> list[float]:
    """
    Convierte una transición en un vector de features para el modelo.
    11 dimensiones — documentadas en el docstring del módulo original.
    """
    from_sal = from_attrs.get("avg_salary",    50_000)
    to_sal   = to_attrs.get("avg_salary",      50_000)
    from_sat = from_attrs.get("satisfaction",  0.7)
    to_sat   = to_attrs.get("satisfaction",    0.7)
    from_dem = from_attrs.get("demand",        0.7)
    to_dem   = to_attrs.get("demand",          0.7)

    return [
        min(to_sal / max(from_sal, 1), 5.0) / 5.0,           # 0: ratio salarial
        to_dem,                                                # 1: demanda destino
        to_sat,                                                # 2: satisfacción destino
        edge_attrs.get("difficulty", 0.5),                     # 3: dificultad
        edge_attrs.get("risk", 0.3),                           # 4: riesgo
        min(edge_attrs.get("transition_years", 2) / 10, 1.0), # 5: años transición
        min(to_attrs.get("years_experience", 3) / 20, 1.0),   # 6: exp. requerida
        to_sat - from_sat,                                     # 7: delta satisfacción
        to_dem - from_dem,                                     # 8: delta demanda
        min((to_sal - from_sal) / max(from_sal, 1), 3.0) / 3, # 9: salary growth
        from_attrs.get("years_experience", 3) / 20,           # 10: exp. acumulada
    ]


# ──────────────────────────────────────────────────────────────
# Generador de datos sintéticos de entrenamiento
# ──────────────────────────────────────────────────────────────

def _generate_training_data(n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera dataset sintético de transiciones de carrera.
    Ruido de label usa uniform(-0.03, 0.03) en lugar de
    normal(0, 0.05) para evitar distribución bimodal en los extremos [0, 1].
    """
    rng = np.random.default_rng(42)
    X_list, y_list = [], []

    for _ in range(n_samples):
        salary_ratio  = rng.uniform(0.8, 4.0) / 5.0
        demand        = rng.beta(5, 2)
        satisfaction  = rng.beta(4, 2)
        difficulty    = rng.beta(2, 3)
        risk          = rng.beta(2, 4)
        trans_years   = rng.uniform(0.1, 1.0)
        exp_required  = rng.uniform(0.0, 0.8)
        delta_sat     = rng.uniform(-0.4, 0.4)
        delta_dem     = rng.uniform(-0.3, 0.3)
        salary_growth = rng.uniform(-0.1, 0.6)
        current_exp   = rng.uniform(0.05, 0.5)

        features = [
            salary_ratio, demand, satisfaction, difficulty, risk,
            trans_years, exp_required, delta_sat, delta_dem,
            salary_growth, current_exp,
        ]

        p_success = (
            0.30 * demand +
            0.25 * satisfaction +
            0.20 * (1 - risk) +
            0.15 * (1 - difficulty) +
            0.10 * max(0.0, delta_sat)
        )
        if salary_ratio > 0.6 and risk > 0.6:
            p_success *= 0.7
        if 0.15 <= salary_ratio <= 0.40 and difficulty < 0.5:
            p_success *= 1.2

        # FIX [S2]: uniform acotado — sin bimodalidad en 0 y 1
        noise     = rng.uniform(-0.03, 0.03)
        p_success = float(np.clip(p_success + noise, 0.0, 1.0))
        y         = int(rng.random() < p_success)

        X_list.append(features)
        y_list.append(y)

    return np.array(X_list), np.array(y_list)


# ──────────────────────────────────────────────────────────────
# Predictor principal
# ──────────────────────────────────────────────────────────────

class CareerOutcomePredictor:
    """
    Predice la probabilidad de éxito de una transición de carrera.
    Modelo: GradientBoostingClassifier (sklearn).
    """

    # FIX [S1]: ruta absoluta + override via env var PATHFORGE_MODEL_PATH
    _DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "career_model.pkl"

    @classmethod
    def _resolve_model_path(cls) -> Path:
        env_path = os.getenv("PATHFORGE_MODEL_PATH")
        if env_path:
            return Path(env_path).resolve()
        return cls._DEFAULT_MODEL_PATH

    def __init__(self) -> None:
        self._pipeline:             Pipeline | None      = None
        self._feature_importances:  dict[str, float]     = {}
        self._cv_score:             float                 = 0.0
        self.MODEL_PATH = self._resolve_model_path()   # FIX [S1]

    # ── Factory ────────────────────────────────────────────────

    @classmethod
    def load_or_train(cls) -> "CareerOutcomePredictor":
        instance = cls()
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn no disponible. Predictor desactivado.")
            return instance

        if instance.MODEL_PATH.exists():
            try:
                instance._load()
                logger.info(f"Modelo cargado desde {instance.MODEL_PATH}")
                return instance
            except Exception as e:
                logger.warning(f"Error cargando modelo: {e}. Reentrenando...")

        instance.train()
        return instance

    # ── Training ───────────────────────────────────────────────

    def train(self, n_samples: int = 2000) -> None:
        if not SKLEARN_AVAILABLE:
            return

        logger.info(f"Entrenando CareerOutcomePredictor con {n_samples} muestras...")
        X, y = _generate_training_data(n_samples)

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                min_samples_leaf=10,
                random_state=42,
            )),
        ])

        self._pipeline.fit(X, y)

        cv_scores    = cross_val_score(self._pipeline, X, y, cv=5, scoring="roc_auc")
        self._cv_score = float(np.mean(cv_scores))

        feat_names = [
            "salary_ratio", "demand_dest", "satisfaction_dest",
            "difficulty", "risk", "transition_years", "exp_required",
            "delta_satisfaction", "delta_demand", "salary_growth", "current_exp",
        ]
        importances = self._pipeline.named_steps["model"].feature_importances_
        self._feature_importances = dict(zip(feat_names, importances.tolist()))

        logger.success(
            f"Modelo entrenado | CV AUC={self._cv_score:.3f} | "
            f"Top feature: {max(self._feature_importances, key=self._feature_importances.get)}"
        )

        try:
            self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._save()
        except Exception as e:
            logger.warning(f"No se pudo guardar el modelo: {e}")

    # ── Prediction ─────────────────────────────────────────────

    def predict_success(
        self,
        from_attrs: dict,
        to_attrs:   dict,
        edge_attrs: dict,
    ) -> float:
        """Predice probabilidad de éxito de la transición [0, 1]."""
        if not SKLEARN_AVAILABLE or self._pipeline is None:
            return 0.5
        features = transition_to_features(from_attrs, to_attrs, edge_attrs)
        prob = float(self._pipeline.predict_proba([features])[0][1])
        return round(prob, 4)

    def predict_trajectory_success(
        self,
        path: tuple[str, ...],
        node_attrs_fn,
        edge_attrs_fn,
    ) -> float:
        """
        Renombrado de score_trajectory() a predict_trajectory_success()
        para evitar colisión semántica con CareerGraph.score_trajectory().

        Calcula el score promedio de una trayectoria completa promediando
        las probabilidades de éxito de cada transición.
        """
        if len(path) < 2:
            return 0.5

        scores = [
            self.predict_success(
                node_attrs_fn(path[i]),
                node_attrs_fn(path[i + 1]),
                edge_attrs_fn(path[i], path[i + 1]),
            )
            for i in range(len(path) - 1)
        ]
        return float(np.mean(scores))

    @property
    def cv_score(self) -> float:
        return self._cv_score

    @property
    def feature_importances(self) -> dict[str, float]:
        return self._feature_importances

    # ── Persistence ────────────────────────────────────────────

    def _save(self) -> None:
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump({
                "pipeline":            self._pipeline,
                "cv_score":            self._cv_score,
                "feature_importances": self._feature_importances,
            }, f)

    def _load(self) -> None:
        with open(self.MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self._pipeline            = data["pipeline"]
        self._cv_score            = data["cv_score"]
        self._feature_importances = data["feature_importances"]
