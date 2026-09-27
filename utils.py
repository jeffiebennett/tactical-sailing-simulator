"""Visualization and race-analysis helpers for the Tactical Sailing Simulator.

Two responsibilities live here:

* Rendering the live overhead race map with Plotly (``render_race_map``).
* Turning a finished (or in-progress) ``RaceState`` into pandas tables for
  the standings board, the decision replay, and a per-boat strategy score
  (``standings_dataframe``, ``decision_log_dataframe``, ``strategy_score``).

Color usage follows a fixed categorical order (each boat keeps the same
color from the map through to the results screen -- color follows the
entity, never its rank) and every marker carries a direct text label so
identity never depends on hue alone.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

import ai
import physics

if TYPE_CHECKING:  # pragma: no cover
    from simulator import Boat, RaceState

# Chart chrome, taken from the validated reference palette (light surface).
_SURFACE = "#fcfcfb"
_PRIMARY_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED_INK = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"


def format_time(total_seconds: float) -> str:
    """Format a duration in seconds as ``M:SS`` (or ``H:MM:SS`` past an hour)."""
    total_seconds = max(0, int(round(total_seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


# --------------------------------------------------------------------------
# Live race map
# --------------------------------------------------------------------------

def render_race_map(race_state: "RaceState") -> go.Figure:
    """Render the overhead race map: course, marks, wind, boats, and trails.

    Args:
        race_state: The race to visualize.

    Returns:
        A Plotly figure ready for ``st.plotly_chart``.
    """
    fig = go.Figure()

    _add_course(fig, race_state)
    _add_trails(fig, race_state)
    _add_boats(fig, race_state)
    _add_wind_indicator(fig, race_state)

    all_x = [m.x for m in race_state.marks] + [b.x for b in race_state.boats] + [0.0]
    all_y = [m.y for m in race_state.marks] + [b.y for b in race_state.boats] + [0.0]
    pad = max(40.0, 0.15 * (max(all_x + all_y) - min(all_x + all_y) + 1e-6))
    x_range = [min(all_x) - pad, max(all_x) + pad]
    y_range = [min(all_y) - pad, max(all_y) + pad]

    fig.update_layout(
        plot_bgcolor=_SURFACE,
        paper_bgcolor=_SURFACE,
        font=dict(color=_PRIMARY_INK, family="system-ui, -apple-system, Segoe UI, sans-serif"),
        xaxis=dict(
            title="meters (east)",
            range=x_range,
            gridcolor=_GRIDLINE,
            zerolinecolor=_BASELINE,
            color=_MUTED_INK,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="meters (north)",
            range=y_range,
            gridcolor=_GRIDLINE,
            zerolinecolor=_BASELINE,
            color=_MUTED_INK,
        ),
        legend=dict(title="Fleet", bgcolor=_SURFACE, bordercolor=_GRIDLINE, borderwidth=1),
        margin=dict(l=10, r=10, t=10, b=10),
        height=560,
    )
    return fig


def _add_course(fig: go.Figure, race_state: "RaceState") -> None:
    xs = [0.0] + [m.x for m in race_state.marks]
    ys = [0.0] + [m.y for m in race_state.marks]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=_BASELINE, width=2, dash="dot"),
            name="Course",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[m.x for m in race_state.marks],
            y=[m.y for m in race_state.marks],
            mode="markers+text",
            marker=dict(symbol="diamond", size=14, color=_SECONDARY_INK),
            text=[m.name for m in race_state.marks],
            textposition="top center",
            textfont=dict(color=_SECONDARY_INK, size=11),
            name="Marks",
            showlegend=False,
            hovertemplate="%{text}<extra></extra>",
        )
    )


def _add_trails(fig: go.Figure, race_state: "RaceState") -> None:
    for boat in race_state.boats:
        if len(boat.trail) < 2:
            continue
        xs, ys = zip(*boat.trail)
        fig.add_trace(
            go.Scatter(
                x=list(xs),
                y=list(ys),
                mode="lines",
                line=dict(color=boat.color, width=2),
                opacity=0.35,
                legendgroup=boat.id,
                showlegend=False,
                hoverinfo="skip",
            )
        )


def _add_boats(fig: go.Figure, race_state: "RaceState") -> None:
    for boat in race_state.boats:
        marker_size = 16 if boat.is_player else 12
        line_width = 2 if boat.is_player else 0
        fig.add_trace(
            go.Scatter(
                x=[boat.x],
                y=[boat.y],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up",
                    size=marker_size,
                    color=boat.color,
                    angle=boat.heading,
                    line=dict(color=_PRIMARY_INK, width=line_width),
                ),
                text=[boat.name],
                textposition="middle right",
                textfont=dict(color=_SECONDARY_INK, size=11),
                legendgroup=boat.id,
                name=boat.name,
                hovertemplate=(
                    f"<b>{boat.name}</b><br>"
                    f"Status: {boat.status.value}<br>"
                    f"Speed: {boat.speed:.1f} kts<br>"
                    f"Heading: {boat.heading:.0f}°<br>"
                    f"Tack: {boat.tack_side}<extra></extra>"
                ),
            )
        )


def _add_wind_indicator(fig: go.Figure, race_state: "RaceState") -> None:
    """Draw a wind arrow + speed readout, anchored in the top-left of the plot area."""
    wind = race_state.wind
    # Arrow points the direction the wind blows TOWARD (opposite of "from").
    # x/y (the arrowhead) are in paper fractions; ax/ay are a pixel offset for
    # the tail, which is the coordinate space add_annotation expects them in.
    toward_rad = math.radians(wind.direction_deg + 180)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.06,
        y=0.94,
        ax=-40 * math.sin(toward_rad),
        ay=40 * math.cos(toward_rad),
        showarrow=True,
        arrowhead=3,
        arrowsize=1.4,
        arrowwidth=2.5,
        arrowcolor=_MUTED_INK,
        text="",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.06,
        y=0.86,
        text=f"Wind {wind.direction_deg:.0f}° @ {wind.speed_kts:.0f} kts",
        showarrow=False,
        font=dict(color=_SECONDARY_INK, size=12),
        bgcolor=_SURFACE,
    )


# --------------------------------------------------------------------------
# Race analysis
# --------------------------------------------------------------------------

def standings_dataframe(race_state: "RaceState") -> pd.DataFrame:
    """Build the live leaderboard as a pandas DataFrame.

    Finished boats are ranked by finish time; boats still racing are
    ranked below them by course progress (marks completed, then
    proximity to the next mark).

    Returns:
        A DataFrame with columns: Rank, Boat, Status, Time/Progress,
        Speed (kts), Tacks, Incidents.
    """
    def sort_key(boat: "Boat"):
        finished = boat.status.value == "finished"
        return (0 if finished else 1, boat.finish_time or 0.0, -ai.progress_metric(boat, race_state))

    ordered = sorted(race_state.boats, key=sort_key)
    rows = []
    for rank, boat in enumerate(ordered, start=1):
        if boat.status.value == "finished":
            progress = format_time(boat.finish_time)
        else:
            marks_left = len(race_state.marks) - boat.next_mark_index
            progress = f"{marks_left} mark(s) to go"
        rows.append(
            {
                "Rank": rank,
                "Boat": boat.name,
                "Status": boat.status.value.title(),
                "Time / Progress": progress,
                "Speed (kts)": round(boat.speed, 1),
                "Tacks": boat.tacks_count,
                "Incidents": boat.incidents_count,
            }
        )
    return pd.DataFrame(rows)


def decision_log_dataframe(race_state: "RaceState") -> pd.DataFrame:
    """Build the full event/decision replay log as a pandas DataFrame.

    Returns:
        A DataFrame with columns: Time, Boat, Category, Event, sorted
        chronologically.
    """
    rows = [
        {
            "Time": format_time(e.time),
            "Boat": e.boat_name,
            "Category": e.category.replace("_", " ").title(),
            "Event": e.message,
        }
        for e in race_state.events
    ]
    return pd.DataFrame(rows)


def strategy_score(boat: "Boat", race_state: "RaceState") -> float:
    """Compute an approximate 0-100 tactical "strategy score" for a boat.

    This is a heuristic composite, not a rigorous VMG-integral analysis:
    it rewards finishing quickly relative to the fastest boat in the
    fleet, and penalizes excess tacks (beyond a rough expected count for
    the course) and incidents (fouls / fumbled roundings / collisions).

    Args:
        boat: The boat to score.
        race_state: The race it competed in.

    Returns:
        A score in [0, 100]; higher is better.
    """
    finishers = [b for b in race_state.boats if b.finish_time is not None]
    expected_tacks = max(1, len(race_state.marks))

    if boat.finish_time is not None and finishers:
        best_time = min(b.finish_time for b in finishers)
        time_score = 100.0 * best_time / boat.finish_time
    else:
        total_marks = len(race_state.marks)
        time_score = 100.0 * (boat.next_mark_index / total_marks) if total_marks else 0.0

    excess_tacks = max(0, boat.tacks_count - expected_tacks)
    clean_score = max(0.0, 100.0 - excess_tacks * 5.0 - boat.incidents_count * 15.0)

    score = 0.6 * time_score + 0.4 * clean_score
    return round(max(0.0, min(100.0, score)), 1)


def strategy_score_table(race_state: "RaceState") -> pd.DataFrame:
    """Compute strategy scores for every boat, sorted best-first.

    Returns:
        A DataFrame with columns: Boat, Strategy Score, Tacks, Incidents,
        Color (the boat's identity color, for chart reuse).
    """
    rows = [
        {
            "Boat": boat.name,
            "Strategy Score": strategy_score(boat, race_state),
            "Tacks": boat.tacks_count,
            "Incidents": boat.incidents_count,
            "Color": boat.color,
        }
        for boat in race_state.boats
    ]
    return pd.DataFrame(rows).sort_values("Strategy Score", ascending=False).reset_index(drop=True)


def render_strategy_chart(race_state: "RaceState") -> go.Figure:
    """Render a horizontal bar chart of strategy scores, one bar per boat.

    Each bar keeps the boat's own identity color (consistent with the
    race map) since this compares named entities, not a magnitude scale.
    """
    table = strategy_score_table(race_state)
    fig = go.Figure(
        go.Bar(
            x=table["Strategy Score"],
            y=table["Boat"],
            orientation="h",
            marker=dict(color=table["Color"]),
            text=table["Strategy Score"],
            texttemplate="%{text:.0f}",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        plot_bgcolor=_SURFACE,
        paper_bgcolor=_SURFACE,
        font=dict(color=_PRIMARY_INK, family="system-ui, -apple-system, Segoe UI, sans-serif"),
        xaxis=dict(title="Strategy Score", range=[0, 105], gridcolor=_GRIDLINE, color=_MUTED_INK),
        yaxis=dict(title=None, color=_MUTED_INK, autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=90 + 40 * len(table),
        showlegend=False,
    )
    return fig
