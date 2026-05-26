"""
core/constraints.py
-------------------
Define y evalúa restricciones sobre trayectorias profesionales.

Separar restricciones del algoritmo de búsqueda permite:
- Agregar/quitar restricciones sin tocar el generador
- Testear cada restricción de forma independiente
- Combinar restricciones con operadores lógicos (AND, OR)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


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
    ) -> bool:
        """
        Evalúa si el camino parcial o completo satisface la restricción.

        Args:
            path: Secuencia de IDs de nodos (puede ser parcial).
            graph: Grafo sobre el que opera.

        Returns:
            True si la restricción se cumple.
        """
        ...

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
    left: Constraint
    right: Constraint

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        return self.left.is_satisfied(path, graph) and self.right.is_satisfied(path, graph)

    def __repr__(self) -> str:
        return f"({self.left} AND {self.right})"


@dataclass
class OrConstraint(Constraint):
    left: Constraint
    right: Constraint

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        return self.left.is_satisfied(path, graph) or self.right.is_satisfied(path, graph)

    def __repr__(self) -> str:
        return f"({self.left} OR {self.right})"


# ---------------------------------------------------------------------------
# Restricciones concretas
# ---------------------------------------------------------------------------

@dataclass
class MaxYearsConstraint(Constraint):
    """La trayectoria no puede superar N años en total."""

    max_years: int

    def is_satisfied(self, path: tuple[str, ...], graph: GraphProtocol) -> bool:
        if len(path) < 2:
            return True
        total = sum(
            graph.edge_attrs(path[i], path[i + 1])["transition_years"]
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
            graph.edge_attrs(path[i], path[i + 1])["risk"]
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
        final_salary = graph.node_attrs(path[-1])["avg_salary"]
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
            graph.edge_attrs(path[i], path[i + 1])["difficulty"]
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
# Fábrica de configuraciones predefinidas
# ---------------------------------------------------------------------------

class ConstraintProfiles:
    """
    Perfiles de restricciones listos para usar en experimentos.
    Cada perfil combina restricciones que representan un tipo de usuario.
    """

    @staticmethod
    def conservative() -> Constraint:
        """Usuario adverso al riesgo, quiere estabilidad."""
        return MaxRiskConstraint(0.35) & MaxDifficultyConstraint(0.55)

    @staticmethod
    def ambitious() -> Constraint:
        """Usuario que prioriza crecimiento salarial rápido."""
        return MinSalaryConstraint(70_000) & MinLengthConstraint(2)

    @staticmethod
    def balanced(max_years: int = 10) -> Constraint:
        """Usuario que busca equilibrio entre riesgo, tiempo y salario."""
        return (
            MaxYearsConstraint(max_years)
            & MaxRiskConstraint(0.50)
            & MinSalaryConstraint(50_000)
        )

    @staticmethod
    def fast_track() -> Constraint:
        """Usuario que quiere llegar lejos en poco tiempo."""
        return MaxYearsConstraint(6) & MinSalaryConstraint(80_000)
