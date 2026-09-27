"""Physics calculations for the Tactical Sailing Simulator.

This module is intentionally free of any Streamlit / simulator-state
dependencies -- it is a pure function library covering:

* VMG (Velocity Made Good) polar performance modeling: how fast a boat
  can sail at a given angle to the wind.
* Tacking maneuver constants and helpers.
* Wind-shadow ("covering") and collision/proximity detection between boats.
* Great-circle-free planar geometry helpers (bearings, distances) used
  by both the AI and the simulator tick loop.

All angles are compass bearings in degrees, where 0 = North, 90 = East,
measured clockwise. Wind direction follows meteorological convention:
it is the bearing the wind is blowing *from*. Positions are plain (x, y)
tuples in meters on a flat local-tangent-plane, with +x = East and
+y = North.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

KNOTS_TO_MPS: float = 0.514444
"""Conversion factor from knots to meters/second."""

TACK_DURATION_S: float = 8.0
"""Time, in seconds, a tacking maneuver takes to complete."""

TACK_SPEED_PENALTY: float = 0.20
"""Fractional speed lost while a boat is actively tacking (0.20 = -20%)."""

NO_GO_ANGLE_DEG: float = 35.0
"""True wind angles tighter than this cannot be sailed (the "no-go zone")."""

OPTIMAL_UPWIND_TWA_DEG: float = 42.0
"""Best true wind angle for upwind (beating) VMG."""

OPTIMAL_DOWNWIND_TWA_DEG: float = 150.0
"""Best true wind angle for downwind (running) VMG."""

COLLISION_RADIUS_M: float = 3.0
"""Hull-to-hull distance, in meters, that constitutes a collision incident."""

COVER_RADIUS_M: float = 25.0
"""Distance, in meters, within which a boat can blanket another's wind."""

COVER_ANGLE_TOLERANCE_DEG: float = 25.0
"""How close (in bearing) the covering boat must sit to upwind of its target."""

COVER_SPEED_PENALTY: float = 0.15
"""Fractional speed lost sailing in another boat's "dirty air"."""

DEFAULT_HULL_SPEED_KTS: float = 10.0
"""Wind speed (kts) above which speed gains taper off (planing/hull-speed cap)."""

DEFAULT_BOAT_SPEED_CONSTANT: float = 0.85
"""Fraction of effective wind speed converted into boat speed at the best angle."""

# Polar performance curve: true wind angle (deg) -> speed factor in [0, 1],
# relative to the boat's best achievable speed in the given breeze. Modeled
# on a generic small single-sail dinghy: unsailable close to head-to-wind,
# fastest on a close/beam reach, easing off a little by dead downwind.
_POLAR_ANGLES = np.array(
    [0, 10, 20, 30, 35, 42, 50, 60, 75, 90, 110, 135, 150, 165, 180], dtype=float
)
_POLAR_FACTORS = np.array(
    [0, 0, 0, 0.05, 0.35, 0.62, 0.70, 0.76, 0.84, 0.90, 0.97, 1.00, 0.97, 0.85, 0.72],
    dtype=float,
)


# --------------------------------------------------------------------------
# Angles & geometry
# --------------------------------------------------------------------------

def normalize_angle(angle_deg: float) -> float:
    """Wrap an angle to the [0, 360) range."""
    return angle_deg % 360.0


def angle_difference(a_deg: float, b_deg: float) -> float:
    """Return the smallest absolute angular difference between two bearings.

    The result is always in [0, 180].
    """
    diff = abs(normalize_angle(a_deg) - normalize_angle(b_deg))
    return diff if diff <= 180.0 else 360.0 - diff


def signed_angle_difference(a_deg: float, b_deg: float) -> float:
    """Return the signed difference (a - b) wrapped to (-180, 180].

    Positive means ``a`` is clockwise of ``b``.
    """
    diff = (normalize_angle(a_deg) - normalize_angle(b_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if diff == -180.0 else diff


def bearing_between(from_xy: Tuple[float, float], to_xy: Tuple[float, float]) -> float:
    """Compute the compass bearing (deg) from one point to another."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if dx == 0 and dy == 0:
        return 0.0
    return normalize_angle(math.degrees(math.atan2(dx, dy)))


def distance_between(a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> float:
    """Euclidean distance, in meters, between two (x, y) points."""
    return float(math.hypot(a_xy[0] - b_xy[0], a_xy[1] - b_xy[1]))


def project_position(
    position: Tuple[float, float], heading_deg: float, speed_kts: float, dt_seconds: float
) -> Tuple[float, float]:
    """Advance a position along a heading at a given speed for a time step.

    Args:
        position: Current (x, y) position in meters.
        heading_deg: Compass heading in degrees.
        speed_kts: Speed over ground in knots.
        dt_seconds: Time step duration in seconds.

    Returns:
        The new (x, y) position in meters.
    """
    speed_mps = speed_kts * KNOTS_TO_MPS
    heading_rad = math.radians(heading_deg)
    dx = speed_mps * math.sin(heading_rad) * dt_seconds
    dy = speed_mps * math.cos(heading_rad) * dt_seconds
    return position[0] + dx, position[1] + dy


# --------------------------------------------------------------------------
# Wind angle & VMG polar
# --------------------------------------------------------------------------

def true_wind_angle(heading_deg: float, wind_from_deg: float) -> float:
    """Compute the true wind angle (TWA) between a heading and the wind's origin bearing.

    Returns a value in [0, 180], where 0 = head-to-wind and 180 = dead downwind.
    """
    return angle_difference(heading_deg, wind_from_deg)


def tack_side(heading_deg: float, wind_from_deg: float) -> str:
    """Determine which tack a boat is on for a given heading and wind direction.

    Returns ``"starboard"`` if the wind is coming over the boat's starboard
    (right) side, otherwise ``"port"``.
    """
    relative = signed_angle_difference(wind_from_deg, heading_deg)
    return "starboard" if relative >= 0 else "port"


def polar_speed_factor(twa_deg: float) -> float:
    """Look up the normalized boat-speed factor for a true wind angle via the polar curve.

    Args:
        twa_deg: True wind angle in degrees (any range; will be normalized to [0, 180]).

    Returns:
        A speed factor in [0, 1].
    """
    twa = abs(twa_deg) % 360.0
    if twa > 180.0:
        twa = 360.0 - twa
    return float(np.interp(twa, _POLAR_ANGLES, _POLAR_FACTORS))


def wind_speed_multiplier(wind_speed_kts: float, hull_speed_kts: float = DEFAULT_HULL_SPEED_KTS) -> float:
    """Model how a boat's speed potential scales with wind strength.

    Light air scales roughly linearly with wind speed. As wind approaches
    (and exceeds) the boat's hull-speed threshold, further gains taper off
    following a square-root curve -- a simplified stand-in for a full
    planing/displacement performance model.

    Args:
        wind_speed_kts: True wind speed in knots.
        hull_speed_kts: The wind speed at which gains begin to taper.

    Returns:
        An effective wind-speed value in knots (never exceeds a gentle
        curve beyond ``hull_speed_kts``).
    """
    if wind_speed_kts <= 0:
        return 0.0
    if wind_speed_kts <= hull_speed_kts:
        return wind_speed_kts
    return hull_speed_kts * math.sqrt(wind_speed_kts / hull_speed_kts)


def boat_speed_kts(
    wind_speed_kts: float,
    heading_deg: float,
    wind_from_deg: float,
    boat_speed_constant: float = DEFAULT_BOAT_SPEED_CONSTANT,
) -> float:
    """Compute a boat's target speed, in knots, for a heading given the current wind.

    Args:
        wind_speed_kts: True wind speed in knots. Must be >= 0.
        heading_deg: The boat's compass heading (0-360).
        wind_from_deg: Compass bearing the wind is blowing from.
        boat_speed_constant: Fraction of "effective wind" converted to boat
            speed at the boat's best angle; tunable per boat class.

    Returns:
        Target boat speed in knots (always >= 0).

    Raises:
        ValueError: If ``wind_speed_kts`` is negative.
    """
    if wind_speed_kts < 0:
        raise ValueError("wind_speed_kts must be >= 0")
    twa = true_wind_angle(heading_deg, wind_from_deg)
    factor = polar_speed_factor(twa)
    effective_wind = wind_speed_multiplier(wind_speed_kts)
    return max(0.0, effective_wind * factor * boat_speed_constant)


def optimal_heading_and_tack(
    position: Tuple[float, float],
    target: Tuple[float, float],
    wind_from_deg: float,
) -> Tuple[float, str]:
    """Compute the best heading (and resulting tack) to make progress toward a target.

    If the direct bearing to the target is sailable (outside the no-go
    zone), sail it directly. Otherwise, pick the close-hauled heading
    (on whichever tack points closer to the target) at the optimal
    upwind VMG angle.

    Args:
        position: Current (x, y) position in meters.
        target: Target (x, y) position in meters (e.g. the next mark).
        wind_from_deg: Compass bearing the wind is blowing from.

    Returns:
        A tuple of (heading_deg, tack) where tack is "port" or "starboard".
    """
    bearing = bearing_between(position, target)
    twa = true_wind_angle(bearing, wind_from_deg)

    if twa >= NO_GO_ANGLE_DEG:
        return bearing, tack_side(bearing, wind_from_deg)

    starboard_heading = normalize_angle(wind_from_deg + OPTIMAL_UPWIND_TWA_DEG)
    port_heading = normalize_angle(wind_from_deg - OPTIMAL_UPWIND_TWA_DEG)

    if angle_difference(starboard_heading, bearing) <= angle_difference(port_heading, bearing):
        return starboard_heading, "starboard"
    return port_heading, "port"


# --------------------------------------------------------------------------
# Proximity: collisions & wind-shadow covering
# --------------------------------------------------------------------------

def find_collisions(
    positions: Sequence[Tuple[str, Tuple[float, float]]],
    radius_m: float = COLLISION_RADIUS_M,
) -> list[Tuple[str, str, float]]:
    """Detect boat pairs closer than the collision radius.

    Args:
        positions: Sequence of (boat_id, (x, y)) pairs.
        radius_m: Distance below which a pair is flagged as a collision.

    Returns:
        A list of (boat_id_a, boat_id_b, distance_m) tuples for every
        pair within the radius.
    """
    incidents: list[Tuple[str, str, float]] = []
    for i in range(len(positions)):
        id_a, pos_a = positions[i]
        for j in range(i + 1, len(positions)):
            id_b, pos_b = positions[j]
            dist = distance_between(pos_a, pos_b)
            if dist <= radius_m:
                incidents.append((id_a, id_b, dist))
    return incidents


def is_covering(
    covering_boat_pos: Tuple[float, float],
    target_boat_pos: Tuple[float, float],
    wind_from_deg: float,
    radius_m: float = COVER_RADIUS_M,
    angle_tolerance_deg: float = COVER_ANGLE_TOLERANCE_DEG,
) -> bool:
    """Determine whether one boat is blanketing another boat's wind.

    A boat covers another when it sits within ``radius_m`` and roughly
    upwind of the target (i.e. the bearing from the target to the
    covering boat is close to the direction the wind is coming from).

    Args:
        covering_boat_pos: (x, y) of the potentially-covering boat.
        target_boat_pos: (x, y) of the boat whose wind might be blanketed.
        wind_from_deg: Compass bearing the wind is blowing from.
        radius_m: Maximum distance for the cover effect to apply.
        angle_tolerance_deg: Maximum bearing deviation from "directly upwind".

    Returns:
        True if ``covering_boat_pos`` is blanketing ``target_boat_pos``.
    """
    dist = distance_between(covering_boat_pos, target_boat_pos)
    if dist > radius_m or dist <= 0:
        return False
    bearing_to_coverer = bearing_between(target_boat_pos, covering_boat_pos)
    return angle_difference(bearing_to_coverer, wind_from_deg) <= angle_tolerance_deg
