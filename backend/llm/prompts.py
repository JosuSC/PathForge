"""
backend/llm/prompts.py
----------------------
Prompts bilingües: inglés interno para mejor interpretación de IA,
respuestas en español para el usuario.

v2 — Fixes:
[PR1] ml_success_prob eliminado de _format_trajectories — nunca existía
      en scores reales, siempre mostraba 50% falso sesgando al LLM.
      Sustituido por transition_probability_score que SÍ viene de graph.py v2.
[PR2] Sección sim_section ahora documentada claramente como opcional.
      Los campos sim_* se usan cuando se integran resultados de /api/simulate.
[PR3] build_ranking_prompt() acepta terminal_groups precalculado para
      evitar recalcular lo que main_api ya hizo.
[PR4] build_terminal_comparison_prompt() ahora se usa desde analyzer.py
      via compare_terminals(). Ya no es función huérfana.
[PR5] _format_trajectories() unificada en inglés — consistente con el
      idioma de los prompts principales para mejor calidad del LLM.
"""

from __future__ import annotations
from backend.core.evaluator import EvaluatedTrajectory


# ──────────────────────────────────────────────────────────────
# Prompts principales
# ──────────────────────────────────────────────────────────────

def build_comparison_prompt(
    trajectories: list[EvaluatedTrajectory],
    user_profile: str = "technology professional",
) -> str:
    traj_text = _format_trajectories(trajectories)
    return f"""You are a world-class career development expert specializing in tech with 20 years of experience.

I'm presenting multiple alternative professional trajectories for a {user_profile}.
Each trajectory starts from the same initial point but leads to DIFFERENT FINAL DESTINATIONS.
There is no single correct endpoint — each destination represents a different type of success.

{traj_text}

Analyze and compare considering:
1. **Alternative Endings**: What kind of success does each final destination represent?
2. **Risk Profile**: Which trajectory is more conservative and which is riskier?
3. **Market Relevance**: Which final roles have higher demand in 2025-2026?
4. **Long-term Satisfaction**: Which trajectory offers greater professional fulfillment?
5. **Recommendation**: Which one do you recommend for this profile and why?
6. **Hidden Risks**: Are there trajectories attractive in numbers but with hidden risks?

**Important**: Respond in SPANISH. Use the numeric data. Structure with 6 numbered points."""


def build_ranking_prompt(
    trajectories: list[EvaluatedTrajectory],
    criterion: str,
    terminal_groups: dict[str, list[EvaluatedTrajectory]] | None = None,  # FIX [PR3]
) -> str:
    """
    FIX [PR3]: acepta terminal_groups precalculado para evitar recalcularlo.
    Si no se pasa, lo calcula internamente (compatibilidad hacia atrás).
    """
    traj_text = _format_trajectories(trajectories)

    # FIX [PR3]: usar el precalculado si viene, calcular si no
    if terminal_groups is None:
        terminal_groups = {}
        for et in trajectories:
            t = et.trajectory.nodes[-1] if et.trajectory.nodes else "unknown"
            terminal_groups.setdefault(t, []).append(et)

    terminal_summary = "\n".join(
        f"  → Final destination '{k}': {len(v)} trajectory/ies lead here"
        for k, v in terminal_groups.items()
    )

    return f"""You are a world-class expert in tech careers and professional decision optimization.

The user has the following objective: "{criterion}"

Final destinations available:
{terminal_summary}

Detailed trajectories:
{traj_text}

Based EXCLUSIVELY on the data and the user's objective:

1. Determine which FINAL DESTINATION aligns best with the objective
2. Rank the trajectories toward that destination from best to worst
3. Mention if any alternative destination deserves consideration
4. Give a clear final recommendation and justification

Format your response in SPANISH as follows:

🎯 Best final destination: [name] — [reason in 1 line]

Ranking toward that destination:
🥇 [Trajectory] — [Reason]
🥈 [Trajectory] — [Reason]
🥉 [Trajectory] — [Reason]

Alternative destination worth considering: [name or "none"] — [reason]

Final recommendation: [your complete recommendation]"""


def build_single_analysis_prompt(
    trajectory: EvaluatedTrajectory,
    user_profile: str = "technology professional",
) -> str:
    """
    FIX [PR2]: sección sim_section claramente documentada como opcional.
    Se activa solo cuando main_api integra resultados de /api/simulate
    en trajectory.scores antes de llamar al analyzer.
    """
    path_str = " → ".join(trajectory.trajectory.nodes)
    s        = trajectory.scores
    terminal = trajectory.trajectory.nodes[-1] if trajectory.trajectory.nodes else "unknown"

    # FIX [PR2]: sim_section se activa con campos reales de simulación
    # Estos campos se añaden a scores cuando se llama a /api/simulate
    # y los resultados se integran en EvaluatedTrajectory.
    sim_section = ""
    if s.get("sim_salary_p50"):
        sim_section = f"""
**Monte Carlo Simulation** (stochastic analysis):
  - Median simulated salary:  ${s.get('sim_salary_p50', 0):,.0f}/year
  - Optimistic salary (p90):  ${s.get('sim_salary_p90', 0):,.0f}/year
  - Average success score:    {s.get('sim_success_mean', 0):.0%}
  - Probability of layoff:    {s.get('sim_risk_layoff', 0):.0%}
  - Probability of burnout:   {s.get('sim_risk_burnout', 0):.0%}"""

    # FIX [PR1]: transition_probability_score reemplaza ml_success_prob
    # transition_probability_score SÍ viene de graph.score_trajectory() v2
    trans_prob = s.get("transition_probability_score")
    trans_line = (
        f"**Avg Transition Probability**: {trans_prob:.0%}"
        if trans_prob is not None
        else ""
    )

    return f"""You are a world-class career development expert specializing in technology.

Conduct an in-depth analysis of the following trajectory for a {user_profile}:

**Trajectory**: {path_str}
**Final Destination**: {terminal}
**Salary Growth**: {s.get('salary_growth', 0):.1%} total increase
**Estimated Final Salary**: ${s.get('final_salary', 0):,.0f}/year
**Average Job Demand**: {s.get('avg_demand', 0):.0%}
**Professional Satisfaction**: {s.get('avg_satisfaction', 0):.0%}
**Total Duration**: {s.get('total_years', 0):.0f} years
**Average Risk**: {s.get('avg_risk', 0):.0%}
**Average Difficulty**: {s.get('avg_difficulty', 0):.0%}
{trans_line}
{sim_section}

Provide:
1. **Executive Summary** (2-3 sentences)
2. **Why this final destination** ({terminal}) is or is not appropriate
3. **Key Strengths** of this trajectory
4. **Main Risks and Challenges**
5. **Critical Skills** to develop at each transition
6. **Final Verdict**: Is it worth it? Who is this ideal for?

**Important**: Respond in SPANISH. Be concrete and actionable."""


def build_terminal_comparison_prompt(
    terminal_groups: dict[str, list[EvaluatedTrajectory]],
    user_profile: str,
    criterion: str,
) -> str:
    """
    FIX [PR4]: ahora se usa en analyzer.compare_terminals().
    Compara el mejor camino hacia cada destino final disponible.
    """
    sections = []
    for terminal, trajs in terminal_groups.items():
        best = trajs[0] if trajs else None
        if not best:
            continue
        s = best.scores
        # FIX [PR1]: transition_probability_score en lugar de ml_success_prob
        trans_prob = s.get("transition_probability_score")
        trans_line = (
            f"  Avg transition prob: {trans_prob:.0%}"
            if trans_prob is not None
            else ""
        )
        sections.append(
            f"**DESTINATION: {terminal.upper()}** ({len(trajs)} trajectory/ies)\n"
            f"  Best trajectory: {' → '.join(best.trajectory.nodes)}\n"
            f"  Final salary:    ${s.get('final_salary', 0):,.0f}/year\n"
            f"  Satisfaction:    {s.get('avg_satisfaction', 0):.0%}\n"
            f"  Risk:            {s.get('avg_risk', 0):.0%}\n"
            f"  Years:           {s.get('total_years', 0):.0f}\n"
            f"  Pareto rank:     {best.pareto_rank}\n"
            f"{trans_line}"
        )

    all_sections = "\n\n".join(sections)

    return f"""You are the world's best career coach, specialized in technology.

A {user_profile} wants to: "{criterion}"

The system has explored ALL possible futures and found {len(terminal_groups)} distinct final destinations:

{all_sections}

Like Doctor Strange analyzing alternative futures:

1. **The best final destination** for this objective and why
2. **Comparison of destinations**: what sacrifices and benefits does each offer
3. **The winning path**: the specific trajectory you would recommend
4. **The other futures**: what would happen if the user chose each alternative
5. **Definitive verdict**: in a single sentence, what should they do

**Important**: Respond in SPANISH. Be dramatic, precise, and use the data."""


# ──────────────────────────────────────────────────────────────
# Helper interno
# ──────────────────────────────────────────────────────────────

def _format_trajectories(trajectories: list[EvaluatedTrajectory]) -> str:
    """
    FIX [PR1]: ml_success_prob eliminado — era siempre 0.5 (falso).
               Reemplazado por transition_probability_score (real).
    FIX [PR5]: etiquetas unificadas en inglés para consistencia con
               el idioma de los prompts principales.
    """
    lines = []
    for i, et in enumerate(trajectories, 1):
        path_str = " → ".join(et.trajectory.nodes)
        s        = et.scores
        terminal = et.trajectory.nodes[-1] if et.trajectory.nodes else "?"

        # FIX [PR1]: transition_probability_score — campo real de graph v2
        trans_prob = s.get("transition_probability_score")
        trans_col  = (
            f"Avg trans. prob: {trans_prob:.0%} | "
            if trans_prob is not None
            else ""
        )

        lines.append(
            f"**Trajectory {i}** [Destination: {terminal}]: {path_str}\n"
            f"  Final salary: ${s.get('final_salary', 0):,.0f}/yr | "
            f"Growth: {s.get('salary_growth', 0):.1%} | "
            f"Demand: {s.get('avg_demand', 0):.0%} | "
            f"Satisfaction: {s.get('avg_satisfaction', 0):.0%} | "
            f"Years: {s.get('total_years', 0):.0f} | "
            f"Risk: {s.get('avg_risk', 0):.0%} | "
            f"{trans_col}"
            f"Pareto rank: {et.pareto_rank}"
        )
    return "\n\n".join(lines)
