"""
core/constraints.py
-------------------
Define y evalúa restricciones sobre trayectorias profesionales.
Incluye restricciones adaptativas por percentil que se ajustan
automáticamente al grafo cargado (careers.json o domain graphs).

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

# FIX [C2]: constante compartida — importada por graph.py y main_api.py también
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
# Operadores lógicos
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
# Restricciones concretas FIJAS (originales — se siguen usando)
# ---------------------------------------------------------------------------

@dataclass
class MaxYearsConstraint(Constraint):
    """La trayectoria no puede superar N años en total."""

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
    """El salario final debe superar un mínimo."""

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
    """La trayectoria debe pasar por un nodo específico."""

    required_node: str

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        return self.required_node in path

    def __repr__(self) -> str:
        return f"RequiredNode({self.required_node})"


# ---------------------------------------------------------------------------
# Restricciones ADAPTATIVAS por percentil (nuevas)
# Se ajustan automáticamente al grafo cargado, sin valores fijos.
# ---------------------------------------------------------------------------

@dataclass
class PercentileSalaryConstraint(Constraint):
    """
    El salario final debe superar un percentil de los salarios del grafo.
    Ej: percentile=0.60 significa que el salario final debe ser mayor
    al 60% de los nodos del grafo.
    """
    percentile: float = 0.50  # 0.0 a 1.0

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        if not path:
            return True
        final_salary = graph.node_attrs(path[-1]).get("avg_salary", 0.0)
        p_key = f"p{int(self.percentile * 100)}"
        thresholds = getattr(graph, '_salary_thresholds', {})
        threshold = thresholds.get(p_key, 0.0)
        return final_salary >= threshold

    def __repr__(self) -> str:
        return f"PercentileSalary({self.percentile:.0%})"


@dataclass
class PercentileRiskConstraint(Constraint):
    """
    El riesgo promedio no debe superar un percentil
    de los riesgos de las aristas del grafo.
    """
    percentile: float = 0.50

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        if len(path) < 2:
            return True
        risks = [
            graph.edge_attrs(path[i], path[i + 1]).get("risk", 0.0)
            for i in range(len(path) - 1)
        ]
        avg_risk = sum(risks) / len(risks)
        p_key = f"p{int(self.percentile * 100)}"
        thresholds = getattr(graph, '_risk_thresholds', {})
        threshold = thresholds.get(p_key, 1.0)
        return avg_risk <= threshold

    def __repr__(self) -> str:
        return f"PercentileRisk({self.percentile:.0%})"


@dataclass
class PercentileDifficultyConstraint(Constraint):
    """
    La dificultad promedio no debe superar un percentil
    de las dificultades de las aristas del grafo.
    """
    percentile: float = 0.50

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        if len(path) < 2:
            return True
        diffs = [
            graph.edge_attrs(path[i], path[i + 1]).get("difficulty", 0.0)
            for i in range(len(path) - 1)
        ]
        avg_diff = sum(diffs) / len(diffs)
        p_key = f"p{int(self.percentile * 100)}"
        thresholds = getattr(graph, '_difficulty_thresholds', {})
        threshold = thresholds.get(p_key, 1.0)
        return avg_diff <= threshold

    def __repr__(self) -> str:
        return f"PercentileDifficulty({self.percentile:.0%})"


# ---------------------------------------------------------------------------
# Fábrica de configuraciones predefinidas — AHORA ADAPTATIVAS
# ---------------------------------------------------------------------------

class ConstraintProfiles:
    """
    Perfiles de restricciones adaptativos al dominio.
    Los umbrales se calculan como percentiles del grafo,
    no como valores fijos. Funcionan igual de bien con
    careers.json que con cualquier domain graph.
    """

    @staticmethod
    def conservative(
        max_risk_percentile: float = 0.40,
        max_difficulty_percentile: float = 0.55,
    ) -> Constraint:
        """
        Usuario adverso al riesgo, quiere estabilidad.
        Solo acepta trayectorias con riesgo y dificultad bajos
        respecto al grafo cargado.
        """
        return (
            PercentileRiskConstraint(max_risk_percentile)
            & PercentileDifficultyConstraint(max_difficulty_percentile)
        )

    @staticmethod
    def ambitious(
        min_salary_percentile: float = 0.60,
        min_steps: int = 2,
    ) -> Constraint:
        """
        Usuario que prioriza crecimiento salarial rápido.
        Exige que el salario final supere el 60% de los nodos del grafo.
        """
        return (
            PercentileSalaryConstraint(min_salary_percentile)
            & MinLengthConstraint(min_steps)
        )

    @staticmethod
    def balanced(
        max_years: int = DEFAULT_MAX_YEARS,
        max_risk_percentile: float = 0.60,
        min_salary_percentile: float = 0.40,
    ) -> Constraint:
        """
        Equilibrio entre tiempo, riesgo y dinero.
        El riesgo y salario se adaptan al grafo.
        """
        return (
            MaxYearsConstraint(max_years)
            & PercentileRiskConstraint(max_risk_percentile)
            & PercentileSalaryConstraint(min_salary_percentile)
        )

    @staticmethod
    def fast_track(
        max_years: int = 6,
        min_salary_percentile: float = 0.70,
    ) -> Constraint:
        """
        Máximo crecimiento en menor tiempo.
        Exige salario alto (percentil 70) en pocos años.
        """
        return (
            MaxYearsConstraint(max_years)
            & PercentileSalaryConstraint(min_salary_percentile)
        )