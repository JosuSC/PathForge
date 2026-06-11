"""
backend/core/simulation.py
--------------------------
Componente de Simulación Estocástica de Carrera Profesional.

"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger


# ──────────────────────────────────────────────────────────────
# Tipos de eventos
# ──────────────────────────────────────────────────────────────

class EventType(str, Enum):
    PROMOTION       = "promotion"
    LAYOFF          = "layoff"
    MARKET_BOOM     = "market_boom"
    MARKET_CRASH    = "market_crash"
    BURNOUT         = "burnout"
    NETWORKING      = "networking"
    SKILL_OBSOLETE  = "skill_obsolete"
    MENTORSHIP      = "mentorship"
    STARTUP_FAILURE = "startup_failure"
    ACQUISITION     = "acquisition"


# FIX [SIM2]: encoding fijo y determinista por EventType
_EVENT_TYPE_ENCODING: dict[str, float] = {
    EventType.PROMOTION.value:       0.0,
    EventType.LAYOFF.value:          0.1,
    EventType.MARKET_BOOM.value:     0.2,
    EventType.MARKET_CRASH.value:    0.3,
    EventType.BURNOUT.value:         0.4,
    EventType.NETWORKING.value:      0.5,
    EventType.SKILL_OBSOLETE.value:  0.6,
    EventType.MENTORSHIP.value:      0.7,
    EventType.STARTUP_FAILURE.value: 0.8,
    EventType.ACQUISITION.value:     0.9,
}


@dataclass
class CareerEvent:
    event_type:  EventType
    node_id:     str
    year:        int
    probability: float
    effects:     dict[str, float]
    description: str = ""

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
    path:               tuple[str, ...]
    events:             list[CareerEvent] = field(default_factory=list)
    final_salary:       float = 0.0
    final_satisfaction: float = 0.0
    effective_years:    float = 0.0
    success_score:      float = 0.0
    reached_terminal:   bool  = True
    terminal_node:      str   = ""

    def to_dict(self) -> dict:
        return {
            "path":               list(self.path),
            "events":             [e.to_dict() for e in self.events],
            "final_salary":       round(self.final_salary, 0),
            "final_satisfaction": round(self.final_satisfaction, 3),
            "effective_years":    round(self.effective_years, 1),
            "success_score":      round(self.success_score, 3),
            "reached_terminal":   self.reached_terminal,
            "terminal_node":      self.terminal_node,
        }


# ──────────────────────────────────────────────────────────────
# Probabilidades por nodo ID (careers.json legacy)
# ──────────────────────────────────────────────────────────────

NODE_EVENT_PROBS: dict[str, dict[EventType, float]] = {
    "junior_dev": {
        EventType.PROMOTION:  0.15, EventType.LAYOFF:     0.08,
        EventType.BURNOUT:    0.12, EventType.MENTORSHIP: 0.20,
        EventType.NETWORKING: 0.25,
    },
    "mid_dev": {
        EventType.PROMOTION:   0.18, EventType.LAYOFF:      0.07,
        EventType.BURNOUT:     0.15, EventType.MARKET_BOOM: 0.20,
        EventType.NETWORKING:  0.22,
    },
    "senior_dev": {
        EventType.PROMOTION:   0.20, EventType.LAYOFF:      0.06,
        EventType.BURNOUT:     0.18, EventType.MARKET_BOOM: 0.25,
        EventType.MENTORSHIP:  0.15,
    },
    "tech_lead": {
        EventType.ACQUISITION: 0.12, EventType.BURNOUT:    0.20,
        EventType.PROMOTION:   0.15, EventType.NETWORKING: 0.30,
    },
    "data_scientist": {
        EventType.MARKET_BOOM:    0.30, EventType.SKILL_OBSOLETE: 0.10,
        EventType.PROMOTION:      0.18, EventType.NETWORKING:     0.22,
    },
    "ml_engineer": {
        EventType.MARKET_BOOM:    0.35, EventType.SKILL_OBSOLETE: 0.08,
        EventType.PROMOTION:      0.20, EventType.ACQUISITION:    0.15,
    },
    "startup_founder": {
        EventType.STARTUP_FAILURE: 0.40, EventType.ACQUISITION:  0.20,
        EventType.MARKET_CRASH:    0.25, EventType.MARKET_BOOM:  0.15,
    },
    "engineering_manager": {
        EventType.LAYOFF:      0.10, EventType.BURNOUT:     0.22,
        EventType.ACQUISITION: 0.15, EventType.NETWORKING:  0.28,
    },
    "cto": {
        EventType.ACQUISITION:  0.20, EventType.BURNOUT:      0.25,
        EventType.MARKET_CRASH: 0.15, EventType.NETWORKING:   0.30,
    },
    "freelancer": {
        EventType.MARKET_CRASH: 0.20, EventType.MARKET_BOOM: 0.25,
        EventType.NETWORKING:   0.30, EventType.BURNOUT:      0.15,
    },
    "product_manager": {
        EventType.PROMOTION:   0.18, EventType.ACQUISITION: 0.14,
        EventType.BURNOUT:     0.16, EventType.NETWORKING:  0.28,
    },
    "devops_engineer": {
        EventType.MARKET_BOOM: 0.28, EventType.BURNOUT:    0.14,
        EventType.PROMOTION:   0.16, EventType.NETWORKING: 0.20,
    },
}

# FIX [SIM1]: fallback por node_type universal — cubre TODOS los domain graphs
NODE_TYPE_EVENT_PROBS: dict[str, dict[EventType, float]] = {
    "entry": {
        EventType.PROMOTION:  0.12, EventType.LAYOFF:     0.09,
        EventType.BURNOUT:    0.10, EventType.MENTORSHIP: 0.18,
        EventType.NETWORKING: 0.20,
    },
    "mid": {
        EventType.PROMOTION:   0.16, EventType.LAYOFF:      0.07,
        EventType.BURNOUT:     0.14, EventType.MARKET_BOOM: 0.18,
        EventType.NETWORKING:  0.22,
    },
    "senior": {
        EventType.PROMOTION:   0.18, EventType.LAYOFF:      0.06,
        EventType.BURNOUT:     0.17, EventType.MARKET_BOOM: 0.22,
        EventType.NETWORKING:  0.25,
    },
    "leadership": {
        EventType.ACQUISITION:  0.15, EventType.BURNOUT:     0.22,
        EventType.MARKET_CRASH: 0.12, EventType.NETWORKING:  0.28,
        EventType.PROMOTION:    0.10,
    },
}

DEFAULT_EVENT_PROBS: dict[EventType, float] = {
    EventType.PROMOTION:  0.10, EventType.LAYOFF:    0.07,
    EventType.BURNOUT:    0.12, EventType.NETWORKING: 0.15,
}

EVENT_EFFECTS: dict[EventType, dict[str, float]] = {
    EventType.PROMOTION:       {"salary_mult": 1.20, "years_reduction":  0.5, "satisfaction_delta": +0.05},
    EventType.LAYOFF:          {"salary_mult": 0.85, "years_increase":   1.0, "satisfaction_delta": -0.10},
    EventType.MARKET_BOOM:     {"salary_mult": 1.15, "difficulty_mult": 0.85, "satisfaction_delta": +0.03},
    EventType.MARKET_CRASH:    {"salary_mult": 0.90, "difficulty_mult": 1.20, "satisfaction_delta": -0.05},
    EventType.BURNOUT:         {"salary_mult": 1.00, "years_increase":   0.5, "satisfaction_delta": -0.15},
    EventType.NETWORKING:      {"salary_mult": 1.05, "difficulty_mult": 0.90, "satisfaction_delta": +0.04},
    EventType.SKILL_OBSOLETE:  {"salary_mult": 0.92, "difficulty_mult": 1.15, "satisfaction_delta": -0.08},
    EventType.MENTORSHIP:      {"salary_mult": 1.08, "difficulty_mult": 0.80, "satisfaction_delta": +0.06},
    EventType.STARTUP_FAILURE: {"salary_mult": 0.70, "years_increase":   2.0, "satisfaction_delta": -0.20},
    EventType.ACQUISITION:     {"salary_mult": 1.30,                          "satisfaction_delta": +0.10},
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

# FIX [SIM4]: límite del multiplicador salarial acumulado
_MAX_SALARY_MULT: float = 2.5
_MIN_SALARY_MULT: float = 0.5

# Eventos negativos para estadísticas de riesgo
_NEGATIVE_EVENTS = {
    EventType.LAYOFF, EventType.BURNOUT,
    EventType.STARTUP_FAILURE, EventType.MARKET_CRASH,
}


# ──────────────────────────────────────────────────────────────
# Simulador principal
# ──────────────────────────────────────────────────────────────

class CareerSimulator:
    """
    Simula la ejecución de una trayectoria profesional con eventos aleatorios.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_simulations: int = 50,
        outcome_model=None,
    ) -> None:
        self._rng          = random.Random(seed)
        self._np_rng       = np.random.default_rng(seed)
        self._n_sims       = n_simulations
        self._outcome_model = outcome_model

    # ── API pública ────────────────────────────────────────────

    def simulate(
        self,
        path:       tuple[str, ...],
        node_attrs: dict[str, dict],
        edge_attrs: dict[tuple, dict],
    ) -> SimulationResult:
        """Simula una trayectoria una vez con eventos estocásticos."""
        if not path:
            return SimulationResult(path=path)

        events:             list[CareerEvent] = []
        salary_mult:        float = 1.0
        satisfaction_delta: float = 0.0
        year_offset:        float = 0.0
        current_year:       int   = 0

        for i, node_id in enumerate(path):
            attrs = node_attrs.get(node_id, {})

            if i > 0:
                edge         = edge_attrs.get((path[i - 1], node_id), {})
                trans_years  = edge.get("transition_years", 2)
                current_year += trans_years

            node_events = self._generate_events(node_id, current_year, attrs)
            events.extend(node_events)

            for ev in node_events:
                eff = ev.effects
                # FIX [SIM4]: acumular y aplicar cap al salary_mult
                salary_mult        = min(
                    _MAX_SALARY_MULT,
                    max(_MIN_SALARY_MULT, salary_mult * eff.get("salary_mult", 1.0))
                )
                satisfaction_delta += eff.get("satisfaction_delta", 0.0)
                year_offset        += eff.get("years_increase",    0.0)
                year_offset        -= eff.get("years_reduction",   0.0)

        terminal_node  = path[-1]
        terminal_attrs = node_attrs.get(terminal_node, {})
        base_sal       = terminal_attrs.get("avg_salary",    50_000)
        base_sat       = terminal_attrs.get("satisfaction",  0.7)

        final_salary       = base_sal * salary_mult
        final_satisfaction = min(1.0, max(0.0, base_sat + satisfaction_delta))
        effective_years    = max(1.0, current_year + year_offset)

        neg_count = sum(1 for e in events if e.event_type in _NEGATIVE_EVENTS)
        success_score = (
            0.35 * min(final_salary / 200_000, 1.0) +
            0.30 * final_satisfaction +
            0.20 * (1.0 - min(effective_years / 20, 1.0)) +
            0.15 * (1.0 - neg_count / max(len(path), 1))
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
        path:       tuple[str, ...],
        node_attrs: dict[str, dict],
        edge_attrs: dict[tuple, dict],
    ) -> dict[str, Any]:
        """
        Ejecuta N simulaciones Monte Carlo de la misma trayectoria.
        """
        # Welford accumulators
        n = 0
        # salary
        sal_mean = sal_m2 = 0.0
        sal_samples: list[float] = []
        # satisfaction
        sat_mean = sat_m2 = 0.0
        # years
        yr_mean = yr_m2 = 0.0
        yr_samples: list[float] = []
        # success
        sc_mean = sc_m2 = 0.0

        best_result:  SimulationResult | None = None
        worst_result: SimulationResult | None = None
        best_score  = -1.0
        worst_score =  2.0

        # Risk event counters
        risk_counts: dict[str, int] = {
            EventType.LAYOFF.value:          0,
            EventType.BURNOUT.value:         0,
            EventType.STARTUP_FAILURE.value: 0,
            EventType.MARKET_CRASH.value:    0,
        }

        for _ in range(self._n_sims):
            r = self.simulate(path, node_attrs, edge_attrs)
            n += 1

            # Welford update
            def _welford(mean: float, m2: float, x: float) -> tuple[float, float]:
                delta  = x - mean
                mean  += delta / n
                m2    += delta * (x - mean)
                return mean, m2

            sal_mean, sal_m2 = _welford(sal_mean, sal_m2, r.final_salary)
            sat_mean, sat_m2 = _welford(sat_mean, sat_m2, r.final_satisfaction)
            yr_mean,  yr_m2  = _welford(yr_mean,  yr_m2,  r.effective_years)
            sc_mean,  sc_m2  = _welford(sc_mean,  sc_m2,  r.success_score)

            sal_samples.append(r.final_salary)
            yr_samples.append(r.effective_years)

            if r.success_score > best_score:
                best_score  = r.success_score
                best_result = r
            if r.success_score < worst_score:
                worst_score  = r.success_score
                worst_result = r

            for ev in r.events:
                if ev.event_type.value in risk_counts:
                    risk_counts[ev.event_type.value] += 1

        sal_arr = np.array(sal_samples)
        yr_arr  = np.array(yr_samples)

        return {
            "n_simulations": self._n_sims,
            "salary": {
                "mean": round(sal_mean, 0),
                "std":  round(float(np.sqrt(sal_m2 / max(n - 1, 1))), 0),
                "p10":  round(float(np.percentile(sal_arr, 10)), 0),
                "p50":  round(float(np.percentile(sal_arr, 50)), 0),
                "p90":  round(float(np.percentile(sal_arr, 90)), 0),
            },
            "satisfaction": {
                "mean": round(sat_mean, 3),
                "std":  round(float(np.sqrt(sat_m2 / max(n - 1, 1))), 3),
            },
            "years": {
                "mean": round(yr_mean, 1),
                "std":  round(float(np.sqrt(yr_m2 / max(n - 1, 1))), 1),
                "p10":  round(float(np.percentile(yr_arr, 10)), 1),
                "p90":  round(float(np.percentile(yr_arr, 90)), 1),
            },
            "success_score": {
                "mean":  round(sc_mean, 3),
                "std":   round(float(np.sqrt(sc_m2 / max(n - 1, 1))), 3),
                "best":  round(best_score,  3),
                "worst": round(worst_score, 3),
            },
            "risk_events": {k: round(v / self._n_sims, 2) for k, v in risk_counts.items()},
            "best_case":   best_result.to_dict()  if best_result  else {},
            "worst_case":  worst_result.to_dict() if worst_result else {},
        }

    # ── Internals ──────────────────────────────────────────────

    def _get_event_probs(
        self, node_id: str, node_attrs: dict
    ) -> dict[EventType, float]:
        """
        Lookup por ID (legacy careers.json) con fallback por
        node_type ("entry"/"mid"/"senior"/"leadership") para domain graphs.
        """
        if node_id in NODE_EVENT_PROBS:
            return NODE_EVENT_PROBS[node_id]

        node_type = node_attrs.get("type", "")
        if node_type in NODE_TYPE_EVENT_PROBS:
            return NODE_TYPE_EVENT_PROBS[node_type]

        return DEFAULT_EVENT_PROBS

    def _generate_events(
        self,
        node_id:    str,
        year:       int,
        node_attrs: dict,
    ) -> list[CareerEvent]:
        """Genera eventos aleatorios para un nodo dado."""
        probs  = self._get_event_probs(node_id, node_attrs)
        events: list[CareerEvent] = []

        for event_type, base_prob in probs.items():
            adj_prob = self._adjust_probability(base_prob, event_type, node_attrs)

            if self._outcome_model:
                features   = self._node_to_features(node_attrs, event_type)
                model_prob = float(self._outcome_model.predict_proba([features])[0][1])
                adj_prob   = 0.6 * adj_prob + 0.4 * model_prob

            if self._rng.random() < adj_prob:
                events.append(CareerEvent(
                    event_type=event_type,
                    node_id=node_id,
                    year=year,
                    probability=adj_prob,
                    effects=dict(EVENT_EFFECTS.get(event_type, {})),
                    description=EVENT_DESCRIPTIONS.get(event_type, ""),
                ))

        return events

    @staticmethod
    def _adjust_probability(
        base_prob:  float,
        event_type: EventType,
        node_attrs: dict,
    ) -> float:
        """Ajusta la probabilidad base según atributos del nodo."""
        p            = base_prob
        satisfaction = node_attrs.get("satisfaction", 0.7)
        demand       = node_attrs.get("demand",       0.7)

        if event_type == EventType.BURNOUT:
            p *= (1.5 - satisfaction)
        elif event_type == EventType.LAYOFF:
            p *= (1.2 - demand * 0.4)
        elif event_type == EventType.MARKET_BOOM:
            p *= (0.5 + demand)
        elif event_type == EventType.MARKET_CRASH:
            p *= (1.2 - demand * 0.3)

        return min(max(p, 0.0), 0.95)

    @staticmethod
    def _node_to_features(node_attrs: dict, event_type: EventType) -> list[float]:
        """
        Encoding fijo y determinista por EventType.
        """
        return [
            node_attrs.get("avg_salary",        50_000) / 200_000,
            node_attrs.get("demand",             0.7),
            node_attrs.get("satisfaction",       0.7),
            node_attrs.get("years_experience",   3) / 20,
            _EVENT_TYPE_ENCODING.get(event_type.value, 0.5),  # FIX [SIM2]
        ]
