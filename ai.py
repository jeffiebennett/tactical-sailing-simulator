"""Competitor AI decision-making for the Tactical Sailing Simulator.

Each non-player boat is assigned a "personality" (conservative, balanced,
or aggressive) at race creation. Personality mainly affects two things:
how often a boat opts to cover the rival just ahead of it instead of
sailing its own optimal VMG course, and (in ``simulator.GameLogic``) its
default mark-rounding style. The actual heading math is delegated to
``physics``; this module only decides *what to aim at*.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import physics

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance for type hints only
    from simulator import Boat, RaceState

_COVER_CHANCE_BY_PERSONALITY = {
    "conservative": 0.05,
    "balanced": 0.15,
    "aggressive": 0.30,
}


def progress_metric(boat: "Boat", race_state: "RaceState") -> float:
    """A larger-is-better proxy for how far along the course a boat is.

    Combines marks completed (dominant term) with proximity to the current
    target mark (tie-breaker), so boats on the same leg can be compared.
    """
    if boat.next_mark_index >= len(race_state.marks):
        return float("inf")
    mark = race_state.marks[boat.next_mark_index]
    dist_to_mark = physics.distance_between(boat.position, mark.position)
    return boat.next_mark_index * 1_000_000.0 - dist_to_mark


def _boat_immediately_ahead(boat: "Boat", race_state: "RaceState") -> Optional["Boat"]:
    """Find the closest rival that is currently ahead of ``boat`` on the course."""
    my_progress = progress_metric(boat, race_state)
    ahead = [
        b
        for b in race_state.boats
        if b.id != boat.id and b.status.value != "finished" and progress_metric(b, race_state) > my_progress
    ]
    if not ahead:
        return None
    return min(ahead, key=lambda b: progress_metric(b, race_state))


def decide_heading(boat: "Boat", race_state: "RaceState") -> Tuple[float, str]:
    """Choose a desired heading and resulting tack for an AI-controlled boat.

    Most of the time the boat simply sails the VMG-optimal course toward
    its next mark. With a personality-dependent probability, it instead
    tries to cover (blanket the wind of) the rival immediately ahead of
    it -- a simple stand-in for real tactical marking.

    Args:
        boat: The AI boat to navigate.
        race_state: The current race state.

    Returns:
        A (heading_deg, tack) tuple, as produced by
        ``physics.optimal_heading_and_tack``.
    """
    wind_from = race_state.wind.direction_deg
    cover_chance = _COVER_CHANCE_BY_PERSONALITY.get(boat.personality, 0.15)

    if cover_chance > 0 and race_state.rng.random() < cover_chance:
        rival = _boat_immediately_ahead(boat, race_state)
        if rival is not None:
            target = physics.project_position(
                rival.position, wind_from, 5.0 / physics.KNOTS_TO_MPS, 1.0
            )
            return physics.optimal_heading_and_tack(boat.position, target, wind_from)

    mark = race_state.marks[boat.next_mark_index]
    return physics.optimal_heading_and_tack(boat.position, mark.position, wind_from)
