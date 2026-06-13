"""
core/constraints.py
-------------------
Define y evalua restricciones sobre trayectorias profesionales.

FIX ADAPTATIVO v2: Las PercentileConstraint ahora tienen "spread awareness" --
si el rango de datos es muy estrecho (ej: riesgo 0.43-0.53), automaticamente
relajan el umbral para no filtrar casi todo. Los perfiles tambien usan
percentiles mas permisivos para funcionar con cualquier dominio.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

# FIX [C2]: constante compartida -- importada por graph.py y main_api.py tambien
DEFAULT_MAX_YEARS: int = 12


# ---------------------------------------------------------------------------
# Protocolo: cualquier objeto que sepa dar atributos de nodo/arista
# ---------------------------------------------------------------------------

class GraphProtocol(Protocol):
    def node_attrs(self, node_id: str) -> dict: ...
    def edge_attrs(self, u: str, v: str) -> dict: ...
    def has_edge(self, u: str, v: str) -> bool: ...


# ---------------------------------------------------------------------------
# Base abstracta
# ---------------------------------------------------------------------------

class Constraint(ABC):
    """Interfaz base para todas las restricciones."""

    @abstractmethod
    def is_satisfied(
        self,
        path: tuple[str, ...],
        graph: GraphProtocol,
    ) -> bool: ...

    def __and__(self, other: "Constraint") -> "AndConstraint":
        return AndConstraint(self, other)

    def __or__(self, other: "Constraint") -> "OrConstraint":
        return OrConstraint(self, other)

    def __repr__(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Operadores logicos
# ---------------------------------------------------------------------------

@dataclass
class AndConstraint(Constraint):
    left:  Constraint
    right: Constraint

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        return self.left.is_satisfied(path, graph) and self.right.is_satisfied(path, graph)

    def __repr__(self) -> str:
        return f"({self.left} AND {self.right})"


@dataclass
class OrConstraint(Constraint):
    left:  Constraint
    right: Constraint

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        return self.left.is_satisfied(path, graph) or self.right.is_satisfied(path, graph)

    def __repr__(self) -> str:
        return f"({self.left} OR {self.right})"


# ---------------------------------------------------------------------------
# Restricciones concretas con valores fijos (legacy, compatibilidad)
# ---------------------------------------------------------------------------

@dataclass
class MaxYearsConstraint(Constraint):
    """La trayectoria no puede superar N anos en total."""

    max_years: int

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        if len(path) < 2:
            return True
        total = sum(
            graph.edge_attrs(path[i], path[i + 1]).get("transition_years", 0)
            for i in range(len(path) - 1)
        )
        return total <= self.max_years

    def __repr__(self) -> str:
        return f"MaxYears({self.max_years})"


@dataclass
class MaxRiskConstraint(Constraint):
    """El riesgo promedio de las transiciones no puede superar el umbral."""

    max_risk: float

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        if len(path) < 2:
            return True
        risks = [
            graph.edge_attrs(path[i], path[i + 1]).get("risk", 0.0)
            for i in range(len(path) - 1)
        ]
        return (sum(risks) / len(risks)) <= self.max_risk

    def __repr__(self) -> str:
        return f"MaxRisk({self.max_risk})"


@dataclass
class MinSalaryConstraint(Constraint):
    """El salario final debe superar un minimo."""

    min_salary: float

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        if not path:
            return True
        final_salary = graph.node_attrs(path[-1]).get("avg_salary", 0.0)
        return final_salary >= self.min_salary

    def __repr__(self) -> str:
        return f"MinSalary({self.min_salary})"


@dataclass
class MinLengthConstraint(Constraint):
    """La trayectoria debe tener al menos N pasos (roles)."""

    min_steps: int

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        return len(path) >= self.min_steps

    def __repr__(self) -> str:
        return f"MinLength({self.min_steps})"


@dataclass
class MaxDifficultyConstraint(Constraint):
    """La dificultad promedio de transiciones no puede superar el umbral."""

    max_difficulty: float

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        if len(path) < 2:
            return True
        diffs = [
            graph.edge_attrs(path[i], path[i + 1]).get("difficulty", 0.0)
            for i in range(len(path) - 1)
        ]
        return (sum(diffs) / len(diffs)) <= self.max_difficulty

    def __repr__(self) -> str:
        return f"MaxDifficulty({self.max_difficulty})"


@dataclass
class RequiredNodeConstraint(Constraint):
    """La trayectoria debe pasar por un nodo especifico."""

    required_node: str

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        return self.required_node in path

    def __repr__(self) -> str:
        return f"RequiredNode({self.required_node})"


# ---------------------------------------------------------------------------
# NUEVAS: Restricciones adaptativas por percentil con "spread awareness"
# ---------------------------------------------------------------------------
# Cuando el rango de datos es muy estrecho (ej: riesgo 0.43-0.53),
# los percentiles producen umbrales muy restrictivos que filtran casi todo.
# Ahora las restricciones detectan rangos estrechos y relajan automaticamente
# el umbral para permitir suficiente variabilidad en las trayectorias.

@dataclass
class PercentileMinSalaryConstraint(Constraint):
    """
    El salario final debe superar el percentil P de los salarios del grafo.

    Spread awareness: si el rango salarial es estrecho (< 20% del max),
    usa un percentil mas bajo automaticamente para no filtrar demasiado.
    """

    percentile: float = 40.0   # percentil 0-100
    _threshold: float | None = None

    def _ensure_threshold(self, graph: GraphProtocol) -> float:
        if self._threshold is not None:
            return self._threshold
        # Intentar usar thresholds pre-computados del grafo
        if hasattr(graph, 'percentile_thresholds') and 'salary' in getattr(graph, 'percentile_thresholds', {}):
            p_key = f"p{int(self.percentile)}"
            sal_info = graph.percentile_thresholds['salary']
            self._threshold = sal_info.get(p_key, 0.0)
            # FIX SPREAD: si el rango es muy estrecho, relajar
            sal_min = sal_info.get('min', 0.0)
            sal_max = sal_info.get('max', 0.0)
            sal_range = sal_max - sal_min
            # Si el rango es < 20% del maximo, usar p10 como minimo
            if sal_max > 0 and sal_range / sal_max < 0.20:
                p10_key = "p10"
                relaxed = sal_info.get(p10_key, self._threshold)
                self._threshold = min(self._threshold, relaxed)
                import numpy as np
                # Usar al menos el minimo absoluto para no filtrar todo
                self._threshold = min(self._threshold, sal_min * 1.01)
        else:
            # Fallback: calcular al vuelo
            salaries = []
            for nid in graph.all_node_ids():
                salaries.append(graph.node_attrs(nid).get("avg_salary", 0.0))
            if not salaries:
                self._threshold = 0.0
            else:
                import numpy as np
                self._threshold = float(np.percentile(salaries, self.percentile))
                # Spread awareness
                sal_min = min(salaries)
                sal_max = max(salaries)
                if sal_max > 0 and (sal_max - sal_min) / sal_max < 0.20:
                    relaxed = float(np.percentile(salaries, 10))
                    self._threshold = min(self._threshold, relaxed)
        return self._threshold

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        if not path:
            return True
        threshold = self._ensure_threshold(graph)
        final_salary = graph.node_attrs(path[-1]).get("avg_salary", 0.0)
        return final_salary >= threshold

    def __repr__(self) -> str:
        return f"PercentileMinSalary(p{int(self.percentile)})"


@dataclass
class PercentileMaxRiskConstraint(Constraint):
    """
    El riesgo promedio no puede superar el percentil P de los riesgos del grafo.

    Spread awareness: si el rango de riesgo es muy estrecho (< 0.15),
    usa el percentil 90 o el maximo como umbral para no filtrar casi todo.
    En rangos estrechos, el riesgo no discrimina bien entre trayectorias.
    """

    percentile: float = 75.0   # percentil 0-100
    _threshold: float | None = None

    # Umbral de spread para riesgo: si (max - min) < esto, relajar
    _RISK_SPREAD_THRESHOLD: float = 0.15

    def _ensure_threshold(self, graph: GraphProtocol) -> float:
        if self._threshold is not None:
            return self._threshold
        if hasattr(graph, 'percentile_thresholds') and 'risk' in getattr(graph, 'percentile_thresholds', {}):
            p_key = f"p{int(self.percentile)}"
            risk_info = graph.percentile_thresholds['risk']
            self._threshold = risk_info.get(p_key, 1.0)
            # FIX SPREAD: si el rango de riesgo es muy estrecho, relajar
            risk_min = risk_info.get('min', 0.0)
            risk_max = risk_info.get('max', 1.0)
            risk_spread = risk_max - risk_min
            if risk_spread < self._RISK_SPREAD_THRESHOLD:
                # Rango estrecho: usar p90 o el maximo como umbral
                p90_val = risk_info.get('p90', risk_max)
                self._threshold = max(self._threshold, p90_val)
                # Ademas anadir un margen: max + 10% del spread
                self._threshold = max(self._threshold, risk_max + risk_spread * 0.10)
                # Pero nunca superar 1.0
                self._threshold = min(self._threshold, 1.0)
        else:
            risks = []
            for nid in graph.all_node_ids():
                for succ in graph.successors(nid):
                    risks.append(graph.edge_attrs(nid, succ).get("risk", 0.0))
            if not risks:
                self._threshold = 1.0
            else:
                import numpy as np
                self._threshold = float(np.percentile(risks, self.percentile))
                risk_min = min(risks)
                risk_max = max(risks)
                if (risk_max - risk_min) < self._RISK_SPREAD_THRESHOLD:
                    relaxed = float(np.percentile(risks, 90))
                    self._threshold = max(self._threshold, relaxed)
        return self._threshold

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        if len(path) < 2:
            return True
        threshold = self._ensure_threshold(graph)
        risks = [
            graph.edge_attrs(path[i], path[i + 1]).get("risk", 0.0)
            for i in range(len(path) - 1)
        ]
        return (sum(risks) / len(risks)) <= threshold

    def __repr__(self) -> str:
        return f"PercentileMaxRisk(p{int(self.percentile)})"


@dataclass
class PercentileMaxDifficultyConstraint(Constraint):
    """
    La dificultad promedio no puede superar el percentil P del grafo.

    Spread awareness: igual que MaxRisk, si el rango es estrecho (< 0.15),
    relaja automaticamente el umbral.
    """

    percentile: float = 75.0   # percentil 0-100
    _threshold: float | None = None

    _DIFF_SPREAD_THRESHOLD: float = 0.15

    def _ensure_threshold(self, graph: GraphProtocol) -> float:
        if self._threshold is not None:
            return self._threshold
        if hasattr(graph, 'percentile_thresholds') and 'difficulty' in getattr(graph, 'percentile_thresholds', {}):
            p_key = f"p{int(self.percentile)}"
            diff_info = graph.percentile_thresholds['difficulty']
            self._threshold = diff_info.get(p_key, 1.0)
            # FIX SPREAD: si el rango de dificultad es muy estrecho, relajar
            diff_min = diff_info.get('min', 0.0)
            diff_max = diff_info.get('max', 1.0)
            diff_spread = diff_max - diff_min
            if diff_spread < self._DIFF_SPREAD_THRESHOLD:
                p90_val = diff_info.get('p90', diff_max)
                self._threshold = max(self._threshold, p90_val)
                self._threshold = max(self._threshold, diff_max + diff_spread * 0.10)
                self._threshold = min(self._threshold, 1.0)
        else:
            diffs = []
            for nid in graph.all_node_ids():
                for succ in graph.successors(nid):
                    diffs.append(graph.edge_attrs(nid, succ).get("difficulty", 0.0))
            if not diffs:
                self._threshold = 1.0
            else:
                import numpy as np
                self._threshold = float(np.percentile(diffs, self.percentile))
                diff_min = min(diffs)
                diff_max = max(diffs)
                if (diff_max - diff_min) < self._DIFF_SPREAD_THRESHOLD:
                    relaxed = float(np.percentile(diffs, 90))
                    self._threshold = max(self._threshold, relaxed)
        return self._threshold

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        if len(path) < 2:
            return True
        threshold = self._ensure_threshold(graph)
        diffs = [
            graph.edge_attrs(path[i], path[i + 1]).get("difficulty", 0.0)
            for i in range(len(path) - 1)
        ]
        return (sum(diffs) / len(diffs)) <= threshold

    def __repr__(self) -> str:
        return f"PercentileMaxDifficulty(p{int(self.percentile)})"


# ---------------------------------------------------------------------------
# Fabrica de configuraciones predefinidas -- ACTUALIZADA con percentiles v2
# ---------------------------------------------------------------------------
# Los percentiles son mas permisivos que antes porque:
# 1. El spread awareness relaja automaticamente en rangos estrechos
# 2. Los percentiles base son mas altos para no filtrar demasiado
# 3. careers.json (rango amplio) produce umbrales similares a los fijos originales

class ConstraintProfiles:
    """Perfiles de restricciones listos para usar en experimentos."""

    @staticmethod
    def conservative() -> Constraint:
        """Usuario adverso al riesgo, quiere estabilidad."""
        return (
            PercentileMaxRiskConstraint(percentile=65.0)
            & PercentileMaxDifficultyConstraint(percentile=75.0)
        )

    @staticmethod
    def ambitious() -> Constraint:
        """Usuario que prioriza crecimiento salarial rapido."""
        return PercentileMinSalaryConstraint(percentile=25.0) & MinLengthConstraint(2)

    @staticmethod
    def balanced(max_years: int = DEFAULT_MAX_YEARS) -> Constraint:
        """Usa DEFAULT_MAX_YEARS=12 sincronizado con el resto del sistema."""
        return (
            MaxYearsConstraint(max_years)
            & PercentileMaxRiskConstraint(percentile=75.0)
            & PercentileMinSalaryConstraint(percentile=20.0)
        )

    @staticmethod
    def fast_track() -> Constraint:
        """Usuario que quiere llegar lejos en poco tiempo."""
        return MaxYearsConstraint(6) & PercentileMinSalaryConstraint(percentile=35.0)
