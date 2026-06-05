"""
Visualizer — creates Plotly charts for evaluation results.
"""

from typing import Dict, List

import plotly.graph_objects as go

from .models import EvalLevel, EvalReport, EvalScore

_LEVEL_COLORS = {
    EvalLevel.SESSION: "#9B59B6",
    EvalLevel.TRACE: "#3498DB",
    EvalLevel.SPAN: "#27AE60",
}


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#4CAF50"
    elif score >= 0.6:
        return "#FF9800"
    return "#F44336"


def create_radar_chart(avg_scores: Dict[str, float]) -> go.Figure:
    """Radar / spider chart of per-evaluator average scores."""
    if not avg_scores:
        return _empty_fig("No scores to display")

    names = list(avg_scores.keys())
    vals = list(avg_scores.values())

    # Shorten long labels for display
    short = [n[:14] + "…" if len(n) > 16 else n for n in names]

    # Close the polygon
    short_plot = short + [short[0]]
    vals_plot = vals + [vals[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=vals_plot,
            theta=short_plot,
            fill="toself",
            fillcolor="rgba(99, 179, 237, 0.20)",
            line=dict(color="#63B3ED", width=2.5),
            marker=dict(size=7, color="#63B3ED"),
            hovertemplate="%{theta}: %{r:.0%}<extra></extra>",
        )
    )

    # Add a 0.6 threshold ring
    fig.add_trace(
        go.Scatterpolar(
            r=[0.6] * (len(short) + 1),
            theta=short_plot,
            mode="lines",
            line=dict(color="rgba(255,165,0,0.4)", width=1.5, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["20%", "40%", "60%", "80%", "100%"],
                gridcolor="rgba(255,255,255,0.12)",
                linecolor="rgba(255,255,255,0.12)",
                tickfont=dict(size=9, color="rgba(200,200,200,0.7)"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.12)",
                linecolor="rgba(255,255,255,0.12)",
                tickfont=dict(size=10, color="rgba(220,220,220,0.85)"),
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        annotations=[
            dict(
                text="— 60% threshold",
                x=0.01,
                y=-0.05,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=10, color="rgba(255,165,0,0.7)"),
            )
        ],
        paper_bgcolor="rgba(25,30,45,0.95)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        height=400,
        margin=dict(l=65, r=65, t=35, b=50),
    )
    return fig


def create_bar_chart(report: EvalReport) -> go.Figure:
    """Horizontal bar chart — one bar per evaluator (avg across targets), color-coded."""
    if not report.scores:
        return _empty_fig("No scores to display")

    avg = report.avg_score_by_evaluator()

    # Map display_name → level for color
    level_map: Dict[str, EvalLevel] = {}
    for s in report.scores:
        level_map[s.evaluator_display] = s.level

    # Sort: SESSION → TRACE → SPAN, then by score descending within each group
    level_order = {EvalLevel.SESSION: 0, EvalLevel.TRACE: 1, EvalLevel.SPAN: 2}
    items = sorted(
        avg.items(),
        key=lambda x: (level_order.get(level_map.get(x[0], EvalLevel.TRACE), 1), -x[1]),
    )

    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    bar_colors = [_score_color(v) for v in vals]
    border_colors = [
        _LEVEL_COLORS.get(level_map.get(n, EvalLevel.TRACE), "#888") for n in names
    ]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color=border_colors, width=2),
            ),
            text=[f"{v:.0%}" for v in vals],
            textposition="outside",
            textfont=dict(color="rgba(230,230,230,0.9)", size=11),
            hovertemplate="%{y}: %{x:.0%}<extra></extra>",
        )
    )

    # Threshold lines
    fig.add_vline(x=0.6, line=dict(color="rgba(255,165,0,0.5)", width=1.5, dash="dot"))
    fig.add_vline(x=0.8, line=dict(color="rgba(76,175,80,0.35)", width=1, dash="dot"))

    fig.update_layout(
        xaxis=dict(
            range=[0, 1.18],
            tickformat=".0%",
            gridcolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="rgba(200,200,200,0.8)"),
        ),
        yaxis=dict(
            tickfont=dict(color="rgba(220,220,220,0.9)", size=11),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        paper_bgcolor="rgba(25,30,45,0.95)",
        plot_bgcolor="rgba(20,25,40,0.5)",
        font=dict(color="white"),
        height=max(280, len(names) * 34 + 70),
        margin=dict(l=20, r=70, t=20, b=50),
        annotations=[
            dict(
                x=0.6,
                y=-0.08,
                xref="x",
                yref="paper",
                text="60%",
                showarrow=False,
                font=dict(size=9, color="rgba(255,165,0,0.8)"),
            ),
            dict(
                x=0.8,
                y=-0.08,
                xref="x",
                yref="paper",
                text="80%",
                showarrow=False,
                font=dict(size=9, color="rgba(76,175,80,0.8)"),
            ),
        ],
    )
    return fig


def create_trace_timeline(report: EvalReport) -> go.Figure:
    """
    Timeline-style heatmap: rows = evaluator names, cols = trace turns.
    Shows how scores evolve across turns.
    """
    if not report.trace_scores:
        return _empty_fig("No trace-level scores to display")

    # Collect unique evaluators and trace IDs (preserving order)
    trace_ids = []
    seen_tids = set()
    for s in report.trace_scores:
        if s.target_id not in seen_tids:
            trace_ids.append(s.target_id)
            seen_tids.add(s.target_id)

    eval_names = []
    seen_evals = set()
    for s in report.trace_scores:
        if s.evaluator_display not in seen_evals:
            eval_names.append(s.evaluator_display)
            seen_evals.add(s.evaluator_display)

    # Build z matrix [eval × trace]
    score_map: Dict[tuple, float] = {
        (s.evaluator_display, s.target_id): s.score for s in report.trace_scores
    }

    z = [[score_map.get((ev, tid), 0.0) for tid in trace_ids] for ev in eval_names]
    text_z = [
        [f"{score_map.get((ev, tid), 0.0):.0%}" for tid in trace_ids]
        for ev in eval_names
    ]

    short_tids = [tid[:8] + "…" if len(tid) > 10 else tid for tid in trace_ids]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"Turn {i + 1}\n({t})" for i, t in enumerate(short_tids)],
            y=eval_names,
            text=text_z,
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            colorscale=[
                [0.0, "#c62828"],
                [0.3, "#e65100"],
                [0.6, "#f9a825"],
                [0.8, "#2e7d32"],
                [1.0, "#1b5e20"],
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(text="Score", font=dict(color="white")),
                tickformat=".0%",
                tickfont=dict(color="white"),
            ),
            hovertemplate="Evaluator: %{y}<br>Turn: %{x}<br>Score: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(25,30,45,0.95)",
        plot_bgcolor="rgba(20,25,40,0.5)",
        font=dict(color="white"),
        xaxis=dict(tickfont=dict(color="rgba(200,200,200,0.9)")),
        yaxis=dict(tickfont=dict(color="rgba(200,200,200,0.9)", size=10)),
        height=max(260, len(eval_names) * 30 + 80),
        margin=dict(l=20, r=60, t=20, b=60),
    )
    return fig


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[
            dict(
                text=message,
                showarrow=False,
                font=dict(size=14, color="rgba(180,180,180,0.7)"),
            )
        ],
        paper_bgcolor="rgba(25,30,45,0.95)",
        plot_bgcolor="rgba(20,25,40,0.5)",
        height=300,
    )
    return fig
