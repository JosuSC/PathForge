"""
backend/llm/prompts.py
----------------------
Prompts bilingües: inglés interno para mejor interpretación de IA,
respuestas en español para el usuario.

Estrategia:
  - Prompt en ENGLISH (mejor interpretación por LLMs)
  - Respuesta en SPANISH (como solicita el usuario)
"""

from __future__ import annotations
from backend.core.evaluator import EvaluatedTrajectory


def build_comparison_prompt(
    trajectories: list[EvaluatedTrajectory],
    user_profile: str = "technology professional",
) -> str:
    traj_text = _format_trajectories(trajectories)
    return f"""You are a world-class career development expert specializing in tech with 20 years of experience.

I'm presenting you with multiple alternative professional trajectories for a {user_profile}.
Each trajectory starts from the same initial point but leads to DIFFERENT FINAL DESTINATIONS.
This is critical: there is no single correct endpoint — each destination represents a different type of success.

{traj_text}

Analyze and compare considering:
1. **Alternative Endings**: What kind of success does each final destination represent? (CTO vs Founder vs Freelancer, etc.)
2. **Risk Profile**: Which trajectory is more conservative and which is riskier?
3. **Market Relevance**: Which final roles have higher demand in 2025-2026?
4. **Long-term Satisfaction**: Which trajectory offers greater professional fulfillment?
5. **Recommendation**: Which one do you recommend for this profile and why?
6. **Hidden Risks**: Are there trajectories attractive in numbers but with hidden risks?

**Important**: Respond in SPANISH. Use the numeric data. Structure with 6 numbered points."""


def build_ranking_prompt(
    trajectories: list[EvaluatedTrajectory],
    criterion: str,
) -> str:
    traj_text = _format_trajectories(trajectories)
    # Group by terminal
    terminals: dict[str, list] = {}
    for et in trajectories:
        t = et.trajectory.nodes[-1] if et.trajectory.nodes else "unknown"
        terminals.setdefault(t, []).append(et)

    terminal_summary = "\n".join(
        f"  → Final destination '{k}': {len(v)} trajectory(ies) lead here"
        for k, v in terminals.items()
    )

    return f"""You are a world-class expert in tech careers and professional decision optimization.

The user has the following objective: "{criterion}"

Final destinations available in this universe of trajectories:
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
    path_str = " → ".join(trajectory.trajectory.nodes)
    s = trajectory.scores
    terminal = trajectory.trajectory.nodes[-1] if trajectory.trajectory.nodes else "unknown"

    sim_section = ""
    if "sim_salary_p50" in s:
        sim_section = f"""
**Monte Carlo Simulation** (stochastic analysis):
  - Median simulated salary: ${s.get('sim_salary_p50', 0):,.0f}/year
  - Optimistic salary (p90):  ${s.get('sim_salary_p90', 0):,.0f}/year
  - Average success score:  {s.get('sim_success_mean', 0):.0%}
  - Probability of layoff:  {s.get('sim_risk_layoff', 0):.0%}
  - Probability of burnout:  {s.get('sim_risk_burnout', 0):.0%}"""

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
**ML Success Probability**: {s.get('ml_success_prob', 0.5):.0%}
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
    """Compares the best trajectories toward each final destination."""
    sections = []
    for terminal, trajs in terminal_groups.items():
        best = trajs[0] if trajs else None
        if not best:
            continue
        s = best.scores
        sections.append(
            f"**DESTINATION: {terminal.upper()}** ({len(trajs)} trajectory/ies)\n"
            f"  Best trajectory: {' → '.join(best.trajectory.nodes)}\n"
            f"  Final salary: ${s.get('final_salary', 0):,.0f}/year\n"
            f"  Satisfaction: {s.get('avg_satisfaction', 0):.0%}\n"
            f"  Risk: {s.get('avg_risk', 0):.0%}\n"
            f"  Years: {s.get('total_years', 0):.0f}\n"
            f"  ML success: {s.get('ml_success_prob', 0.5):.0%}"
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


def _format_trajectories(trajectories: list[EvaluatedTrajectory]) -> str:
    lines = []
    for i, et in enumerate(trajectories, 1):
        path_str = " → ".join(et.trajectory.nodes)
        s = et.scores
        terminal = et.trajectory.nodes[-1] if et.trajectory.nodes else "?"
        lines.append(
            f"**Trayectoria {i}** [Final: {terminal}]: {path_str}\n"
            f"  Salario final: ${s.get('final_salary', 0):,.0f}/año | "
            f"Crecimiento: {s.get('salary_growth', 0):.1%} | "
            f"Demanda: {s.get('avg_demand', 0):.0%} | "
            f"Satisfacción: {s.get('avg_satisfaction', 0):.0%} | "
            f"Años: {s.get('total_years', 0):.0f} | "
            f"Riesgo: {s.get('avg_risk', 0):.0%} | "
            f"ML éxito: {s.get('ml_success_prob', 0.5):.0%} | "
            f"Rank Pareto: {et.pareto_rank}"
        )
    return "\n\n".join(lines)