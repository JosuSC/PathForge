"""
backend/core/simulation.py
--------------------------
Componente de Simulación Estocástica de Carrera Profesional.

Modela eventos aleatorios que ocurren durante una trayectoria:
- Ascensos acelerados (rendimiento excepcional)
- Despidos / downsizing (crisis económica)
- Cambios del mercado (nueva tecnología emerge)
- Burnout (satisfacción baja → reduce velocidad)
- Networking efectivo (reduce dificultad de transición)

Integración con IA:
- Los pesos de probabilidad de eventos se aprenden desde datos
  históricos usando sklearn (GradientBoostingClassifier)
- Cada simulación genera un "mundo posible" diferente

Esto convierte el problema de optimización en un MDP parcialmente
observable (POMDP), añadiendo el componente de simulación requerido.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from loguru import logger


# ──────────────────────────────────────────────────────────────
# Tipos de eventos
# ──────────────────────────────────────────────────────────────

class EventType(str, Enum):
    PROMOTION       = "promotion"        # Ascenso acelerado
    LAYOFF          = "layoff"           # Despido / reducción
    MARKET_BOOM     = "market_boom"      # Alta demanda del sector
    MARKET_CRASH    = "market_crash"     # Recesión / baja demanda
    BURNOUT         = "burnout"          # Agotamiento profesional
    NETWORKING      = "networking"       # Conexiones valiosas
    SKILL_OBSOLETE  = "skill_obsolete"   # Habilidad obsoleta
    MENTORSHIP      = "mentorship"       # Mentor clave encontrado
    STARTUP_FAILURE = "startup_failure"  # Fracaso emprendimiento
    ACQUISITION     = "acquisition"      # Empresa adquirida


@dataclass
class CareerEvent:
    """Representa un evento que ocurre en un paso de la trayectoria."""

    event_type:    EventType
    node_id:       str                    # Nodo donde ocurrió
    year:          int                    # Año de ocurrencia
    probability:   float                  # Probabilidad con que ocurrió
    effects:       dict[str, float]       # Modificadores aplicados
    description:   str = ""              # Descripción para el LLM

    def to_dict(self) -> dict:
        return {
            "type":        self.event_type.value,
            "node":        self.node_id,
            "year":        self.year,
            "probability": round(self.probability, 3),
            "effects":     self.effects,
            "description": self.description,
        }


@dataclass
class SimulationResult:
    """
    Resultado de simular una trayectoria con eventos estocásticos.
    Cada simulación del mismo camino puede dar resultados distintos.
    """

    path:              tuple[str, ...]
    events:            list[CareerEvent] = field(default_factory=list)
    final_salary:      float = 0.0
    final_satisfaction: float = 0.0
    effective_years:   float = 0.0       # Puede variar por ascensos/despidos
    success_score:     float = 0.0       # Score compuesto 0-1
    reached_terminal:  bool = True       # ¿Llegó al nodo final?
    terminal_node:     str = ""

    def to_dict(self) -> dict:
        return {
            "path":              list(self.path),
            "events":            [e.to_dict() for e in self.events],
            "final_salary":      round(self.final_salary, 0),
            "final_satisfaction": round(self.final_satisfaction, 3),
            "effective_years":   round(self.effective_years, 1),
            "success_score":     round(self.success_score, 3),
            "reached_terminal":  self.reached_terminal,
            "terminal_node":     self.terminal_node,
        }


# ──────────────────────────────────────────────────────────────
# Definición de eventos por tipo de nodo
# ──────────────────────────────────────────────────────────────

# Base probabilities per event at each node type
# (calibrated from tech industry data)
NODE_EVENT_PROBS: dict[str, dict[EventType, float]] = {
    "junior_dev": {
        EventType.PROMOTION:   0.15,
        EventType.LAYOFF:      0.08,
        EventType.BURNOUT:     0.12,
        EventType.MENTORSHIP:  0.20,
        EventType.NETWORKING:  0.25,
    },
    "mid_dev": {
        EventType.PROMOTION:   0.18,
        EventType.LAYOFF:      0.07,
        EventType.BURNOUT:     0.15,
        EventType.MARKET_BOOM: 0.20,
        EventType.NETWORKING:  0.22,
    },
    "senior_dev": {
        EventType.PROMOTION:   0.20,
        EventType.LAYOFF:      0.06,
        EventType.BURNOUT:     0.18,
        EventType.MARKET_BOOM: 0.25,
        EventType.MENTORSHIP:  0.15,
    },
    "tech_lead": {
        EventType.ACQUISITION: 0.12,
        EventType.BURNOUT:     0.20,
        EventType.PROMOTION:   0.15,
        EventType.NETWORKING:  0.30,
    },
    "data_scientist": {
        EventType.MARKET_BOOM:    0.30,
        EventType.SKILL_OBSOLETE: 0.10,
        EventType.PROMOTION:      0.18,
        EventType.NETWORKING:     0.22,
    },
    "ml_engineer": {
        EventType.MARKET_BOOM:    0.35,
        EventType.SKILL_OBSOLETE: 0.08,
        EventType.PROMOTION:      0.20,
        EventType.ACQUISITION:    0.15,
    },
    "startup_founder": {
        EventType.STARTUP_FAILURE: 0.40,
        EventType.ACQUISITION:     0.20,
        EventType.MARKET_CRASH:    0.25,
        EventType.MARKET_BOOM:     0.15,
    },
    "engineering_manager": {
        EventType.LAYOFF:      0.10,
        EventType.BURNOUT:     0.22,
        EventType.ACQUISITION: 0.15,
        EventType.NETWORKING:  0.28,
    },
    "cto": {
        EventType.ACQUISITION:  0.20,
        EventType.BURNOUT:      0.25,
        EventType.MARKET_CRASH: 0.15,
        EventType.NETWORKING:   0.30,
    },
    "freelancer": {
        EventType.MARKET_CRASH:  0.20,
        EventType.MARKET_BOOM:   0.25,
        EventType.NETWORKING:    0.30,
        EventType.BURNOUT:       0.15,
    },
    "product_manager": {
        EventType.PROMOTION:   0.18,
        EventType.ACQUISITION: 0.14,
        EventType.BURNOUT:     0.16,
        EventType.NETWORKING:  0.28,
    },
    "devops_engineer": {
        EventType.MARKET_BOOM: 0.28,
        EventType.BURNOUT:     0.14,
        EventType.PROMOTION:   0.16,
        EventType.NETWORKING:  0.20,
    },
}

# Default for unknown nodes
DEFAULT_EVENT_PROBS: dict[EventType, float] = {
    EventType.PROMOTION:   0.10,
    EventType.LAYOFF:      0.07,
    EventType.BURNOUT:     0.12,
    EventType.NETWORKING:  0.15,
}

# Effects of each event on trajectory attributes
EVENT_EFFECTS: dict[EventType, dict[str, float]] = {
    EventType.PROMOTION:       {"salary_mult": 1.20, "years_reduction": 0.5, "satisfaction_delta": +0.05},
    EventType.LAYOFF:          {"salary_mult": 0.85, "years_increase":  1.0, "satisfaction_delta": -0.10},
    EventType.MARKET_BOOM:     {"salary_mult": 1.15, "difficulty_mult": 0.85, "satisfaction_delta": +0.03},
    EventType.MARKET_CRASH:    {"salary_mult": 0.90, "difficulty_mult": 1.20, "satisfaction_delta": -0.05},
    EventType.BURNOUT:         {"salary_mult": 1.00, "years_increase":  0.5, "satisfaction_delta": -0.15},
    EventType.NETWORKING:      {"salary_mult": 1.05, "difficulty_mult": 0.90, "satisfaction_delta": +0.04},
    EventType.SKILL_OBSOLETE:  {"salary_mult": 0.92, "difficulty_mult": 1.15, "satisfaction_delta": -0.08},
    EventType.MENTORSHIP:      {"salary_mult": 1.08, "difficulty_mult": 0.80, "satisfaction_delta": +0.06},
    EventType.STARTUP_FAILURE: {"salary_mult": 0.70, "years_increase":  2.0, "satisfaction_delta": -0.20},
    EventType.ACQUISITION:     {"salary_mult": 1.30, "satisfaction_delta": +0.10},
}

EVENT_DESCRIPTIONS: dict[EventType, str] = {
    EventType.PROMOTION:       "Ascenso acelerado por rendimiento excepcional",
    EventType.LAYOFF:          "Despido por reducción de plantilla / reestructuración",
    EventType.MARKET_BOOM:     "Boom del mercado tecnológico — alta demanda de tu perfil",
    EventType.MARKET_CRASH:    "Recesión económica — mercado más competitivo",
    EventType.BURNOUT:         "Burnout profesional — necesidad de pausar o reorientar",
    EventType.NETWORKING:      "Conexiones clave que abren nuevas puertas",
    EventType.SKILL_OBSOLETE:  "Tecnología dominada queda obsoleta",
    EventType.MENTORSHIP:      "Mentor clave acelera tu desarrollo",
    EventType.STARTUP_FAILURE: "El emprendimiento fracasa — reorientación necesaria",
    EventType.ACQUISITION:     "La empresa es adquirida — oportunidades y cambios",
}


# ──────────────────────────────────────────────────────────────
# Simulador principal
# ──────────────────────────────────────────────────────────────

class CareerSimulator:
    """
    Simula la ejecución de una trayectoria profesional con eventos aleatorios.

    Modelo:
        Para cada nodo en la trayectoria, genera eventos estocásticos
        basados en probabilidades calibradas por tipo de rol.
        Los eventos modifican salario, tiempo y satisfacción.

    Conexión con sklearn:
        Si se provee un modelo entrenado (CareerOutcomePredictor),
        las probabilidades base se ajustan con predicciones del modelo.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_simulations: int = 50,
        outcome_model=None,         # CareerOutcomePredictor (opcional)
    ) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._n_sims = n_simulations
        self._outcome_model = outcome_model

    # ── API pública ────────────────────────────────────────────

    def simulate(
        self,
        path: tuple[str, ...],
        node_attrs: dict[str, dict],     # nodeId → attrs dict
        edge_attrs: dict[tuple, dict],   # (u,v) → attrs dict
    ) -> SimulationResult:
        """
        Simula una trayectoria una vez con eventos estocásticos.

        Args:
            path:       Secuencia de IDs de nodos.
            node_attrs: Atributos de cada nodo.
            edge_attrs: Atributos de cada arista.

        Returns:
            SimulationResult con eventos ocurridos y métricas finales.
        """
        if len(path) < 1:
            return SimulationResult(path=path)

        events: list[CareerEvent] = []
        salary_mult = 1.0
        satisfaction_delta = 0.0
        year_offset = 0.0
        current_year = 0

        for i, node_id in enumerate(path):
            attrs = node_attrs.get(node_id, {})
            base_salary = attrs.get("avg_salary", 50000)
            base_satisfaction = attrs.get("satisfaction", 0.7)

            # Transición (arista)
            if i > 0:
                edge = edge_attrs.get((path[i-1], node_id), {})
                trans_years = edge.get("transition_years", 2)
                current_year += trans_years

            # Generar eventos para este nodo
            node_events = self._generate_events(node_id, current_year, attrs)
            events.extend(node_events)

            # Aplicar efectos de eventos
            for ev in node_events:
                eff = ev.effects
                salary_mult         *= eff.get("salary_mult", 1.0)
                satisfaction_delta  += eff.get("satisfaction_delta", 0.0)
                year_offset         += eff.get("years_increase", 0.0)
                year_offset         -= eff.get("years_reduction", 0.0)

        # Calcular métricas finales
        terminal_node   = path[-1]
        terminal_attrs  = node_attrs.get(terminal_node, {})
        base_sal        = terminal_attrs.get("avg_salary", 50000)
        base_sat        = terminal_attrs.get("satisfaction", 0.7)

        final_salary        = base_sal * salary_mult
        final_satisfaction  = min(1.0, max(0.0, base_sat + satisfaction_delta))
        effective_years     = max(1.0, current_year + year_offset)

        # Success score compuesto (para ranking)
        success_score = (
            0.35 * min(final_salary / 200000, 1.0) +
            0.30 * final_satisfaction +
            0.20 * (1.0 - min(effective_years / 20, 1.0)) +
            0.15 * (1.0 - sum(1 for e in events if e.event_type in {
                EventType.LAYOFF, EventType.BURNOUT, EventType.STARTUP_FAILURE
            }) / max(len(path), 1))
        )

        return SimulationResult(
            path=path,
            events=events,
            final_salary=final_salary,
            final_satisfaction=final_satisfaction,
            effective_years=effective_years,
            success_score=success_score,
            reached_terminal=True,
            terminal_node=terminal_node,
        )

    def monte_carlo(
        self,
        path: tuple[str, ...],
        node_attrs: dict[str, dict],
        edge_attrs: dict[tuple, dict],
    ) -> dict[str, Any]:
        """
        Ejecuta N simulaciones de la misma trayectoria (Monte Carlo).
        Retorna estadísticas agregadas: media, percentiles, peor/mejor caso.

        Esto implementa el componente de SIMULACIÓN del proyecto.
        """
        results = [
            self.simulate(path, node_attrs, edge_attrs)
            for _ in range(self._n_sims)
        ]

        salaries   = [r.final_salary      for r in results]
        sats       = [r.final_satisfaction for r in results]
        years      = [r.effective_years    for r in results]
        scores     = [r.success_score      for r in results]

        # Frecuencia de eventos negativos
        neg_events = {
            EventType.LAYOFF.value:          0,
            EventType.BURNOUT.value:         0,
            EventType.STARTUP_FAILURE.value: 0,
            EventType.MARKET_CRASH.value:    0,
        }
        for r in results:
            for ev in r.events:
                if ev.event_type.value in neg_events:
                    neg_events[ev.event_type.value] += 1

        return {
            "n_simulations": self._n_sims,
            "salary": {
                "mean":   round(float(np.mean(salaries)), 0),
                "p10":    round(float(np.percentile(salaries, 10)), 0),
                "p50":    round(float(np.percentile(salaries, 50)), 0),
                "p90":    round(float(np.percentile(salaries, 90)), 0),
            },
            "satisfaction": {
                "mean":   round(float(np.mean(sats)), 3),
                "std":    round(float(np.std(sats)), 3),
            },
            "years": {
                "mean":   round(float(np.mean(years)), 1),
                "p10":    round(float(np.percentile(years, 10)), 1),
                "p90":    round(float(np.percentile(years, 90)), 1),
            },
            "success_score": {
                "mean":   round(float(np.mean(scores)), 3),
                "std":    round(float(np.std(scores)), 3),
                "best":   round(float(np.max(scores)), 3),
                "worst":  round(float(np.min(scores)), 3),
            },
            "risk_events": {k: round(v / self._n_sims, 2) for k, v in neg_events.items()},
            "best_case":   results[int(np.argmax(scores))].to_dict(),
            "worst_case":  results[int(np.argmin(scores))].to_dict(),
        }

    # ── Internals ──────────────────────────────────────────────

    def _generate_events(
        self,
        node_id: str,
        year: int,
        node_attrs: dict,
    ) -> list[CareerEvent]:
        """
        Genera eventos aleatorios para un nodo dado.
        Usa probabilidades calibradas por tipo de nodo.
        """
        probs = NODE_EVENT_PROBS.get(node_id, DEFAULT_EVENT_PROBS)
        events: list[CareerEvent] = []

        for event_type, base_prob in probs.items():
            # Ajuste por atributos del nodo
            adj_prob = self._adjust_probability(base_prob, event_type, node_attrs)

            # Ajuste por modelo sklearn si disponible
            if self._outcome_model:
                features = self._node_to_features(node_attrs, event_type)
                model_prob = float(self._outcome_model.predict_proba([features])[0][1])
                adj_prob = 0.6 * adj_prob + 0.4 * model_prob

            if self._rng.random() < adj_prob:
                effects = dict(EVENT_EFFECTS.get(event_type, {}))
                events.append(CareerEvent(
                    event_type=event_type,
                    node_id=node_id,
                    year=year,
                    probability=adj_prob,
                    effects=effects,
                    description=EVENT_DESCRIPTIONS.get(event_type, ""),
                ))

        return events

    @staticmethod
    def _adjust_probability(
        base_prob: float,
        event_type: EventType,
        node_attrs: dict,
    ) -> float:
        """Ajusta la probabilidad base según atributos del nodo."""
        p = base_prob
        satisfaction = node_attrs.get("satisfaction", 0.7)
        demand       = node_attrs.get("demand", 0.7)

        # Alta satisfacción reduce burnout y layoffs
        if event_type == EventType.BURNOUT:
            p *= (1.5 - satisfaction)
        elif event_type == EventType.LAYOFF:
            p *= (1.2 - demand * 0.4)
        # Alta demanda aumenta boom y reduce crash
        elif event_type == EventType.MARKET_BOOM:
            p *= (0.5 + demand)
        elif event_type == EventType.MARKET_CRASH:
            p *= (1.2 - demand * 0.3)

        return min(max(p, 0.0), 0.95)

    @staticmethod
    def _node_to_features(node_attrs: dict, event_type: EventType) -> list[float]:
        """Convierte atributos del nodo en feature vector para sklearn."""
        return [
            node_attrs.get("avg_salary", 50000) / 200000,
            node_attrs.get("demand", 0.7),
            node_attrs.get("satisfaction", 0.7),
            node_attrs.get("years_experience", 3) / 20,
            hash(event_type.value) % 10 / 10,  # event type encoding
        ]