"""Core simulation state for the Tactical Sailing Simulator.

Defines the data model (``Boat``, ``WindField``, ``RaceMark``,
``RaceState``) and the ``GameLogic`` engine that advances the race one
10-second tick at a time, applies player decisions, and evaluates the
win condition. Pure math (VMG, tacking geometry, collisions) lives in
``physics``; competitor decision-making lives in ``ai``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import ai
import physics

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

TICK_SECONDS: float = 10.0
TACK_TRIGGER_DEADBAND_DEG: float = 5.0
"""Minimum heading change required to count as needing a real tack maneuver."""
MIN_TACK_INTERVAL_S: float = 25.0
"""Autopilot/AI tack lockout: real crews commit to a tack rather than re-litigating
every tick, so a boat sailing on autopilot won't consider tacking again this soon
after its last one. Manual player tacks (the Tack/Jibe button) bypass this --
they're already costed by the normal tack time/speed penalty."""
MAX_TRAIL_POINTS: int = 60
MAX_RACE_TIME_S: float = 3600.0
"""Safety cap: any boat still racing after this long is scored as DNF."""
TIGHT_ROUNDING_FUMBLE_CHANCE: float = 0.20
TIGHT_ROUNDING_FUMBLE_SPEED_MULT: float = 0.40
TIGHT_ROUNDING_GAIN_M: float = 5.0
WIDE_ROUNDING_SPEED_MULT: float = 0.90
COLLISION_PENALTY_SPEED_MULT: float = 0.50

ROUNDING_STYLES = ("tight", "standard", "wide")
SAILING_STYLES = ("conservative", "balanced", "aggressive")


class BoatStatus(str, Enum):
    """Lifecycle state of a boat during a race."""

    SAILING = "sailing"
    TACKING = "tacking"
    FINISHED = "finished"


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class RaceMark:
    """A fixed course mark (buoy) boats must round in order."""

    id: str
    name: str
    x: float
    y: float
    radius: float = 25.0
    """Distance, in meters, within which a boat is considered to have rounded the mark."""

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class RaceEvent:
    """A single logged occurrence, used for the post-race decision replay."""

    time: float
    boat_id: str
    boat_name: str
    category: str
    message: str


@dataclass
class Boat:
    """A single racing dinghy, player-controlled or AI-controlled.

    Position is stored in meters on a flat local plane (+x East, +y North);
    heading is a compass bearing in degrees.
    """

    id: str
    name: str
    x: float
    y: float
    heading: float
    color: str
    is_player: bool = False
    personality: str = "balanced"  # AI flavor: "conservative" | "balanced" | "aggressive"

    status: BoatStatus = BoatStatus.SAILING
    speed: float = 0.0
    tack_side: str = "starboard"
    tack_time_remaining: float = 0.0
    tack_start_heading: float = 0.0
    target_heading: float = 0.0
    last_tack_time: float = float("-inf")

    next_mark_index: int = 0
    pending_rounding_style: Optional[str] = None
    cover_target_id: Optional[str] = None
    sailing_style: str = "balanced"

    distance_traveled: float = 0.0
    tacks_count: int = 0
    incidents_count: int = 0
    finish_time: Optional[float] = None

    trail: list = field(default_factory=list)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @position.setter
    def position(self, value: Tuple[float, float]) -> None:
        self.x, self.y = value

    def record_trail_point(self) -> None:
        """Append the current position to the trail, trimming old history."""
        self.trail.append(self.position)
        if len(self.trail) > MAX_TRAIL_POINTS:
            self.trail.pop(0)

    def set_sailing_style(self, style: str) -> None:
        """Set the boat's risk posture, used by autopilot/AI heading choices.

        Raises:
            ValueError: If ``style`` is not a recognized sailing style.
        """
        if style not in SAILING_STYLES:
            raise ValueError(f"Unknown sailing style {style!r}; expected one of {SAILING_STYLES}")
        self.sailing_style = style

    def set_rounding_style(self, style: str) -> None:
        """Queue a mark-rounding approach for the next mark this boat reaches.

        Raises:
            ValueError: If ``style`` is not a recognized rounding style.
        """
        if style not in ROUNDING_STYLES:
            raise ValueError(f"Unknown rounding style {style!r}; expected one of {ROUNDING_STYLES}")
        self.pending_rounding_style = style


@dataclass
class WindField:
    """Models true wind direction/speed and periodic random shifts."""

    base_direction_deg: float
    base_speed_kts: float
    direction_deg: float = 0.0
    speed_kts: float = 0.0
    next_shift_at: float = 0.0
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.direction_deg = self.base_direction_deg
        self.speed_kts = self.base_speed_kts

    def schedule_next_shift(self, elapsed_time: float, rng: random.Random) -> None:
        """Pick when (in elapsed race seconds) the next wind shift will occur."""
        self.next_shift_at = elapsed_time + rng.uniform(30.0, 60.0)

    def maybe_shift(self, elapsed_time: float, rng: random.Random) -> Optional[float]:
        """Apply a random directional shift if it is due.

        Args:
            elapsed_time: Current race time in seconds.
            rng: Shared random source (kept on the RaceState for reproducibility).

        Returns:
            The signed shift in degrees if a shift occurred this call, else None.
        """
        if elapsed_time < self.next_shift_at:
            return None
        shift = rng.uniform(-10.0, 10.0)
        self.direction_deg = physics.normalize_angle(self.direction_deg + shift)
        self.history.append((elapsed_time, self.direction_deg, self.speed_kts))
        self.schedule_next_shift(elapsed_time, rng)
        return shift


@dataclass
class RaceState:
    """The complete, mutable state of a single race."""

    boats: list
    wind: WindField
    marks: list
    tick_seconds: float = TICK_SECONDS
    elapsed_time: float = 0.0
    status: str = "setup"  # "setup" | "running" | "finished"
    events: list = field(default_factory=list)
    finish_order: list = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)

    def player_boat(self) -> Boat:
        """Return the single player-controlled boat.

        Raises:
            ValueError: If no player boat exists in this race.
        """
        for boat in self.boats:
            if boat.is_player:
                return boat
        raise ValueError("RaceState has no player boat")

    def boat_by_id(self, boat_id: str) -> Optional[Boat]:
        return next((b for b in self.boats if b.id == boat_id), None)

    def active_boats(self) -> list:
        return [b for b in self.boats if b.status != BoatStatus.FINISHED]

    def log_event(self, boat: Boat, category: str, message: str) -> None:
        self.events.append(RaceEvent(self.elapsed_time, boat.id, boat.name, category, message))


# --------------------------------------------------------------------------
# Game engine
# --------------------------------------------------------------------------

# Fixed categorical order (never cycled/reassigned by rank) so each boat keeps
# one identity color from the race map through to the results screen. Slot 1
# (blue) is reserved for the player. Capped at 8 -- see MAX_COMPETITORS.
_BOAT_COLORS = [
    "#2a78d6",  # 1 blue   (player)
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
MAX_COMPETITORS: int = len(_BOAT_COLORS) - 1
"""Cap on AI competitors, keeping total boats within the validated 8-color palette."""
_PERSONALITIES = ("conservative", "balanced", "aggressive")


class GameLogic:
    """Stateless engine that creates races, advances ticks, and applies decisions.

    All methods operate on a ``RaceState`` passed in explicitly; the class
    holds no instance state of its own.
    """

    @staticmethod
    def create_race(
        num_competitors: int,
        wind_speed_kts: float,
        wind_direction_deg: float,
        race_distance_m: float = 800.0,
        course_type: str = "windward_leeward",
        seed: Optional[int] = None,
    ) -> RaceState:
        """Build a new, ready-to-run ``RaceState``.

        Args:
            num_competitors: Number of AI-controlled boats (player is added on top).
            wind_speed_kts: Starting true wind speed in knots.
            wind_direction_deg: Compass bearing the wind blows from at race start.
            race_distance_m: Approximate length of the first beat, in meters.
            course_type: "windward_leeward" or "triangle".
            seed: Optional RNG seed for reproducible races.

        Returns:
            A populated ``RaceState`` with status "running".

        Raises:
            ValueError: If ``num_competitors`` is negative or wind speed is negative.
        """
        if num_competitors < 0:
            raise ValueError("num_competitors must be >= 0")
        if num_competitors > MAX_COMPETITORS:
            raise ValueError(
                f"num_competitors must be <= {MAX_COMPETITORS} "
                "(keeps every boat on a distinct, colorblind-safe color)"
            )
        if wind_speed_kts < 0:
            raise ValueError("wind_speed_kts must be >= 0")

        rng = random.Random(seed)
        wind_direction_deg = physics.normalize_angle(wind_direction_deg)
        wind = WindField(base_direction_deg=wind_direction_deg, base_speed_kts=wind_speed_kts)
        wind.schedule_next_shift(0.0, rng)

        marks = GameLogic._build_course(course_type, wind_direction_deg, race_distance_m)

        boats: list = []
        num_boats = num_competitors + 1
        start_spacing = 12.0
        start_x0 = -(num_boats - 1) * start_spacing / 2.0
        # Boats line up on a start line perpendicular to the wind, all facing
        # upwind toward the first mark on a sailable close-hauled heading.
        start_heading, start_tack = physics.optimal_heading_and_tack(
            (0.0, 0.0), marks[0].position, wind_direction_deg
        )

        for i in range(num_boats):
            is_player = i == 0
            boat = Boat(
                id="player" if is_player else f"cpu{i}",
                name="You" if is_player else f"Rival {chr(ord('A') + i - 1)}",
                x=start_x0 + i * start_spacing,
                y=0.0,
                heading=start_heading,
                color=_BOAT_COLORS[i % len(_BOAT_COLORS)],
                is_player=is_player,
                personality="balanced" if is_player else rng.choice(_PERSONALITIES),
                tack_side=start_tack,
            )
            boat.record_trail_point()
            boats.append(boat)

        race_state = RaceState(boats=boats, wind=wind, marks=marks, rng=rng, status="running")
        return race_state

    @staticmethod
    def _build_course(course_type: str, wind_direction_deg: float, distance_m: float) -> list:
        """Lay out course marks for the requested course type."""
        origin = (0.0, 0.0)

        def mark_at(bearing_deg: float, dist: float, mark_id: str, name: str) -> RaceMark:
            rad = math.radians(bearing_deg)
            return RaceMark(mark_id, name, origin[0] + dist * math.sin(rad), origin[1] + dist * math.cos(rad))

        if course_type == "triangle":
            leg = distance_m * 0.8
            return [
                mark_at(wind_direction_deg, leg, "mark1", "Windward Mark"),
                mark_at(physics.normalize_angle(wind_direction_deg + 120), leg, "mark2", "Wing Mark"),
                mark_at(physics.normalize_angle(wind_direction_deg + 240), leg, "mark3", "Leeward Mark"),
                RaceMark("finish", "Finish Line", 0.0, 0.0, radius=20.0),
            ]

        # Default: windward-leeward. Sail straight upwind, then straight back.
        return [
            mark_at(wind_direction_deg, distance_m, "mark1", "Windward Mark"),
            RaceMark("finish", "Finish Line", 0.0, 0.0, radius=20.0),
        ]

    # -- Player decisions ----------------------------------------------

    @staticmethod
    def request_tack(race_state: RaceState) -> None:
        """Immediately begin a tack/jibe for the player boat, if able.

        No-op if the player boat is already tacking or has finished.
        """
        boat = race_state.player_boat()
        if boat.status != BoatStatus.SAILING:
            return
        new_tack = "port" if boat.tack_side == "starboard" else "starboard"
        # Mirror the current heading across the wind axis to land on the
        # equivalent angle on the opposite tack (works for beats and runs).
        mirrored = physics.normalize_angle(2 * race_state.wind.direction_deg - boat.heading)
        GameLogic._begin_tack(race_state, boat, mirrored, new_tack)

    @staticmethod
    def set_rounding_style(race_state: RaceState, style: str) -> None:
        """Queue the player's approach style for the next mark."""
        race_state.player_boat().set_rounding_style(style)

    @staticmethod
    def set_sailing_style(race_state: RaceState, style: str) -> None:
        """Set the player's overall risk posture (conservative/balanced/aggressive)."""
        race_state.player_boat().set_sailing_style(style)

    @staticmethod
    def set_cover_target(race_state: RaceState, target_id: Optional[str]) -> None:
        """Set (or clear, with ``None``) which competitor the player is covering."""
        boat = race_state.player_boat()
        if target_id is not None and race_state.boat_by_id(target_id) is None:
            raise ValueError(f"Unknown boat id {target_id!r}")
        boat.cover_target_id = target_id

    # -- Decision context (for the UI) -----------------------------------

    @staticmethod
    def get_decision_context(race_state: RaceState) -> dict:
        """Summarize the tactical situation the player is currently facing.

        Returns a dict with keys: ``mark`` (next RaceMark), ``distance_to_mark``,
        ``approaching_mark`` (bool, within 8x mark radius), ``nearby_competitors``
        (list of (boat_id, name, distance_m) within cover range, nearest first),
        ``wind_shift_deg`` (signed shift this tick, or None), ``current_tack``.
        """
        boat = race_state.player_boat()
        context: dict = {
            "mark": None,
            "distance_to_mark": None,
            "approaching_mark": False,
            "nearby_competitors": [],
            "current_tack": boat.tack_side,
            "status": boat.status.value,
        }
        if boat.status == BoatStatus.FINISHED:
            return context

        mark = race_state.marks[boat.next_mark_index]
        dist = physics.distance_between(boat.position, mark.position)
        context["mark"] = mark
        context["distance_to_mark"] = dist
        context["approaching_mark"] = dist <= mark.radius * 8

        nearby = []
        for other in race_state.boats:
            if other.id == boat.id or other.status == BoatStatus.FINISHED:
                continue
            d = physics.distance_between(boat.position, other.position)
            if d <= physics.COVER_RADIUS_M * 1.5:
                nearby.append((other.id, other.name, d))
        nearby.sort(key=lambda t: t[2])
        context["nearby_competitors"] = nearby

        return context

    # -- Tick / physics --------------------------------------------------

    @staticmethod
    def tick(race_state: RaceState) -> RaceState:
        """Advance the race by one time step (``race_state.tick_seconds``).

        Applies wind evolution, AI + player-autopilot navigation, the
        tacking state machine, mark rounding, and collision detection,
        then evaluates the win/finish condition. Mutates and returns
        ``race_state``; a no-op if the race is not currently running.
        """
        if race_state.status != "running":
            return race_state

        dt = race_state.tick_seconds
        wind = race_state.wind

        shift = wind.maybe_shift(race_state.elapsed_time, race_state.rng)
        if shift is not None:
            direction = "right" if shift > 0 else "left"
            race_state.log_event(
                race_state.boats[0],
                "wind_shift",
                f"Wind shifted {abs(shift):.0f}° {direction} to {wind.direction_deg:.0f}°",
            )

        for boat in race_state.active_boats():
            desired_heading, desired_tack = GameLogic._navigate(boat, race_state)
            GameLogic._advance_boat(boat, race_state, desired_heading, desired_tack, dt)

        GameLogic._apply_covering(race_state)
        GameLogic._check_mark_roundings(race_state)
        GameLogic._check_collisions(race_state)

        race_state.elapsed_time += dt
        GameLogic._check_finish(race_state)
        return race_state

    @staticmethod
    def _navigate(boat: Boat, race_state: RaceState) -> Tuple[float, str]:
        """Compute a boat's desired heading/tack this tick (autopilot or AI)."""
        if not boat.is_player:
            return ai.decide_heading(boat, race_state)

        target_pos = boat.position
        if boat.cover_target_id is not None:
            rival = race_state.boat_by_id(boat.cover_target_id)
            if rival is not None and rival.status != BoatStatus.FINISHED:
                # Steer toward a point just upwind of the rival to blanket their air.
                upwind_bearing = race_state.wind.direction_deg
                target_pos = physics.project_position(
                    rival.position, upwind_bearing, 5.0 / physics.KNOTS_TO_MPS, 1.0
                )
                return physics.optimal_heading_and_tack(boat.position, target_pos, race_state.wind.direction_deg)

        mark = race_state.marks[boat.next_mark_index]
        return physics.optimal_heading_and_tack(boat.position, mark.position, race_state.wind.direction_deg)

    @staticmethod
    def _begin_tack(race_state: RaceState, boat: Boat, target_heading: float, new_tack: str) -> None:
        boat.status = BoatStatus.TACKING
        boat.tack_start_heading = boat.heading
        boat.target_heading = target_heading
        boat.tack_time_remaining = physics.TACK_DURATION_S
        boat.tack_side = new_tack
        boat.tacks_count += 1
        boat.last_tack_time = race_state.elapsed_time
        race_state.log_event(boat, "tack", f"{boat.name} tacks onto {new_tack}")

    @staticmethod
    def _advance_boat(
        boat: Boat, race_state: RaceState, desired_heading: float, desired_tack: str, dt: float
    ) -> None:
        wind = race_state.wind

        if boat.status == BoatStatus.TACKING:
            boat.tack_time_remaining -= dt
            progress = 1.0 - max(boat.tack_time_remaining, 0.0) / physics.TACK_DURATION_S
            swing = physics.signed_angle_difference(boat.target_heading, boat.tack_start_heading)
            boat.heading = physics.normalize_angle(boat.tack_start_heading + swing * progress)
            speed = physics.boat_speed_kts(wind.speed_kts, boat.heading, wind.direction_deg)
            speed *= 1.0 - physics.TACK_SPEED_PENALTY
            if boat.tack_time_remaining <= 0:
                boat.status = BoatStatus.SAILING
                boat.heading = boat.target_heading
                boat.tack_time_remaining = 0.0
        else:
            off_lockout = (race_state.elapsed_time - boat.last_tack_time) >= MIN_TACK_INTERVAL_S
            needs_tack = (
                desired_tack != boat.tack_side
                and physics.angle_difference(boat.heading, desired_heading) > TACK_TRIGGER_DEADBAND_DEG
                and off_lockout
            )
            if needs_tack:
                GameLogic._begin_tack(race_state, boat, desired_heading, desired_tack)
                boat.tack_time_remaining -= dt
                progress = 1.0 - max(boat.tack_time_remaining, 0.0) / physics.TACK_DURATION_S
                swing = physics.signed_angle_difference(boat.target_heading, boat.tack_start_heading)
                boat.heading = physics.normalize_angle(boat.tack_start_heading + swing * progress)
                speed = physics.boat_speed_kts(wind.speed_kts, boat.heading, wind.direction_deg)
                speed *= 1.0 - physics.TACK_SPEED_PENALTY
            else:
                boat.heading = desired_heading
                speed = physics.boat_speed_kts(wind.speed_kts, boat.heading, wind.direction_deg)

        boat.speed = speed
        old_pos = boat.position
        boat.position = physics.project_position(old_pos, boat.heading, speed, dt)
        boat.distance_traveled += physics.distance_between(old_pos, boat.position)
        boat.record_trail_point()

    @staticmethod
    def _apply_covering(race_state: RaceState) -> None:
        wind_from = race_state.wind.direction_deg
        active = race_state.active_boats()
        for boat in active:
            for other in active:
                if other.id == boat.id:
                    continue
                if physics.is_covering(other.position, boat.position, wind_from):
                    boat.speed *= 1.0 - physics.COVER_SPEED_PENALTY
                    break

    @staticmethod
    def _check_mark_roundings(race_state: RaceState) -> None:
        for boat in race_state.active_boats():
            mark = race_state.marks[boat.next_mark_index]
            if physics.distance_between(boat.position, mark.position) > mark.radius:
                continue

            style = boat.pending_rounding_style or (
                "tight" if boat.personality == "aggressive" and not boat.is_player else
                "wide" if boat.personality == "conservative" and not boat.is_player else
                "standard"
            )

            if style == "tight":
                if race_state.rng.random() < TIGHT_ROUNDING_FUMBLE_CHANCE:
                    boat.speed *= TIGHT_ROUNDING_FUMBLE_SPEED_MULT
                    boat.incidents_count += 1
                    race_state.log_event(boat, "incident", f"{boat.name} fumbles a tight rounding of {mark.name}")
                else:
                    next_target = (
                        race_state.marks[boat.next_mark_index + 1]
                        if boat.next_mark_index + 1 < len(race_state.marks)
                        else mark
                    )
                    nudge_bearing = physics.bearing_between(mark.position, next_target.position)
                    boat.position = physics.project_position(
                        boat.position, nudge_bearing, TIGHT_ROUNDING_GAIN_M / physics.KNOTS_TO_MPS, 1.0
                    )
                    race_state.log_event(boat, "mark_rounding", f"{boat.name} nails a tight rounding of {mark.name}")
            elif style == "wide":
                boat.speed *= WIDE_ROUNDING_SPEED_MULT
                race_state.log_event(boat, "mark_rounding", f"{boat.name} takes a safe, wide line around {mark.name}")
            else:
                race_state.log_event(boat, "mark_rounding", f"{boat.name} rounds {mark.name}")

            boat.pending_rounding_style = None
            boat.next_mark_index += 1

            if boat.next_mark_index >= len(race_state.marks):
                boat.status = BoatStatus.FINISHED
                boat.finish_time = race_state.elapsed_time
                race_state.finish_order.append(boat.id)
                race_state.log_event(boat, "finish", f"{boat.name} finishes the race!")

    @staticmethod
    def _check_collisions(race_state: RaceState) -> None:
        active = race_state.active_boats()
        positions = [(b.id, b.position) for b in active]
        for id_a, id_b, dist in physics.find_collisions(positions):
            boat_a = race_state.boat_by_id(id_a)
            boat_b = race_state.boat_by_id(id_b)
            if boat_a is None or boat_b is None:
                continue
            boat_a.speed *= COLLISION_PENALTY_SPEED_MULT
            boat_b.speed *= COLLISION_PENALTY_SPEED_MULT
            boat_a.incidents_count += 1
            boat_b.incidents_count += 1
            race_state.log_event(
                boat_a, "collision", f"{boat_a.name} and {boat_b.name} have a close encounter ({dist:.1f}m)!"
            )
            # Physically separate the pair so boats sailing near-parallel courses
            # don't keep re-triggering a "collision" every subsequent tick.
            dx, dy = boat_b.x - boat_a.x, boat_b.y - boat_a.y
            if dist > 1e-6:
                ux, uy = dx / dist, dy / dist
            else:
                ux, uy = 1.0, 0.0
            push = (physics.COLLISION_RADIUS_M - dist) / 2.0 + 0.5
            boat_a.x, boat_a.y = boat_a.x - ux * push, boat_a.y - uy * push
            boat_b.x, boat_b.y = boat_b.x + ux * push, boat_b.y + uy * push

    @staticmethod
    def _check_finish(race_state: RaceState) -> None:
        all_finished = all(b.status == BoatStatus.FINISHED for b in race_state.boats)
        timed_out = race_state.elapsed_time >= MAX_RACE_TIME_S
        if all_finished or timed_out:
            if timed_out:
                for boat in race_state.active_boats():
                    race_state.log_event(boat, "dnf", f"{boat.name} did not finish (time limit)")
            race_state.status = "finished"

    @staticmethod
    def run_autopilot(race_state: RaceState, max_ticks: int = 400) -> RaceState:
        """Advance the race automatically (including the player boat) until it finishes.

        Useful for a "simulate to finish" fast-forward. The player boat is
        driven by the same autopilot logic used between manual decisions,
        so any cover target / sailing style / rounding preference already
        set continues to apply.

        Args:
            race_state: The race to advance.
            max_ticks: Safety cap on the number of ticks to run.

        Returns:
            The same ``race_state``, advanced in place.
        """
        ticks = 0
        while race_state.status == "running" and ticks < max_ticks:
            GameLogic.tick(race_state)
            ticks += 1
        return race_state
