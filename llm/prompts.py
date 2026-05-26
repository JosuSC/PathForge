"""
llm/prompts.py
--------------
Prompts centralizados para análisis de trayectorias con Gemini.

Principios de prompt engineering aplicados:
- Rol explícito (system-like instruction al inicio)
- Formato de salida especificado
- Ejemplos implícitos en la estructura
- Instrucciones negativas para evitar alucinaciones
"""

from __future__ import annotations

from core.evaluator import EvaluatedTrajectory


def build_comparison_prompt(
    trajectories: list[EvaluatedTrajectory],
    user_profile: str = "profesional de tecnología",
) -> str:
    """
    Construye un prompt para que Gemini compare cualitativamente
    un conjunto de trayectorias del frente de Pareto.

    Args:
        trajectories: Lista de trayectorias evaluadas (idealmente el frente Pareto).
        user_profile: Descripción del perfil del usuario.

    Returns:
        Prompt listo para enviar a Gemini.
    """
    trajectories_text = _format_trajectories(trajectories)

    return f"""Eres un experto en desarrollo de carrera profesional en tecnología con 20 años de experiencia como career coach en empresas de Silicon Valley, Europa y Latinoamérica.

Se te presentan varias trayectorias profesionales alternativas para un {user_profile}. Cada una representa un trade-off óptimo (frente de Pareto) entre múltiples objetivos: crecimiento salarial, demanda laboral, satisfacción profesional, tiempo y riesgo.

{trajectories_text}

Analiza y compara estas trayectorias cualitativamente considerando:

1. **Perfil de riesgo**: ¿Qué trayectoria es más conservadora? ¿Cuál es más arriesgada y por qué?
2. **Contexto del mercado actual**: ¿Cuáles roles tienen mayor relevancia en el mercado tecnológico de 2025-2026?
3. **Satisfacción a largo plazo**: ¿Qué trayectoria ofrece mayor realización profesional y por qué?
4. **Recomendación personalizada**: Según los datos, ¿cuál recomendarías y para qué tipo de persona?
5. **Alertas o consideraciones**: ¿Hay alguna trayectoria que parezca atractiva en números pero tenga riesgos ocultos?

Responde en español. Sé específico, usa los datos numéricos para respaldar tus argumentos. No inventes información que no esté en los datos. Estructura tu respuesta con los 5 puntos numerados."""


def build_single_analysis_prompt(
    trajectory: EvaluatedTrajectory,
    user_profile: str = "profesional de tecnología",
) -> str:
    """
    Prompt para análisis profundo de una sola trayectoria.
    Útil para cuando el usuario quiere explorar una en detalle.
    """
    path_str = " → ".join(trajectory.trajectory.nodes)
    scores = trajectory.scores

    return f"""Eres un experto en desarrollo de carrera profesional en tecnología.

Analiza en profundidad la siguiente trayectoria profesional para un {user_profile}:

**Trayectoria**: {path_str}
**Crecimiento salarial**: {scores.get('salary_growth', 0):.1%} de aumento total
**Salario final estimado**: ${scores.get('final_salary', 0):,.0f}/año
**Demanda laboral promedio**: {scores.get('avg_demand', 0):.0%}
**Satisfacción profesional promedio**: {scores.get('avg_satisfaction', 0):.0%}
**Duración total estimada**: {scores.get('total_years', 0):.0f} años
**Riesgo promedio de transición**: {scores.get('avg_risk', 0):.0%}
**Dificultad promedio**: {scores.get('avg_difficulty', 0):.0%}

Proporciona:
1. **Resumen ejecutivo** (2-3 oraciones)
2. **Puntos fuertes** de esta trayectoria
3. **Desafíos principales** que enfrentará quien la siga
4. **Habilidades clave** que debe desarrollar en cada transición
5. **Veredicto final**: ¿Vale la pena? ¿Para quién es ideal?

Responde en español. Sé concreto y accionable."""


def build_ranking_prompt(
    trajectories: list[EvaluatedTrajectory],
    criterion: str,
) -> str:
    """
    Prompt para que Gemini rankee trayectorias según un criterio específico
    expresado en lenguaje natural.

    Args:
        trajectories: Trayectorias a rankear.
        criterion: Criterio en lenguaje natural, ej: "quiero ganar mucho pero rápido"
    """
    trajectories_text = _format_trajectories(trajectories)

    return f"""Eres un experto en carreras tecnológicas y optimización de decisiones profesionales.

El usuario tiene el siguiente objetivo: "{criterion}"

Aquí están las trayectorias disponibles:

{trajectories_text}

Basándote EXCLUSIVAMENTE en los datos numéricos proporcionados y en el objetivo del usuario:

1. Rankea las trayectorias de mejor a peor para este objetivo específico.
2. Para cada una, explica en 1-2 oraciones por qué está en esa posición.
3. Indica cuál es tu recomendación final y por qué.

Formato de respuesta:
🥇 [Trayectoria] — [Razón]
🥈 [Trayectoria] — [Razón]
🥉 [Trayectoria] — [Razón]
...

Recomendación final: [Tu recomendación con justificación]

Responde en español."""


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------

def _format_trajectories(trajectories: list[EvaluatedTrajectory]) -> str:
    """Formatea una lista de trayectorias como texto estructurado para el prompt."""
    lines: list[str] = []

    for i, et in enumerate(trajectories, 1):
        path_str = " → ".join(et.trajectory.nodes)
        s = et.scores
        lines.append(
            f"**Trayectoria {i}**: {path_str}\n"
            f"  - Crecimiento salarial: {s.get('salary_growth', 0):.1%}\n"
            f"  - Salario final: ${s.get('final_salary', 0):,.0f}/año\n"
            f"  - Demanda laboral: {s.get('avg_demand', 0):.0%}\n"
            f"  - Satisfacción: {s.get('avg_satisfaction', 0):.0%}\n"
            f"  - Años totales: {s.get('total_years', 0):.0f}\n"
            f"  - Riesgo: {s.get('avg_risk', 0):.0%}\n"
            f"  - Dificultad: {s.get('avg_difficulty', 0):.0%}\n"
            f"  - Rank Pareto: {et.pareto_rank}\n"
        )

    return "\n".join(lines)
