"""Tactical Sailing Simulator -- Streamlit entry point.

Run with:

    streamlit run main.py

The race is turn-based: each click of "Advance 10s" (or "Simulate to
Finish") steps the physics forward, driven by whatever tack/cover/style/
rounding decisions are currently set in the control panel. Because
Streamlit reruns this script top-to-bottom on every interaction, the
live ``RaceState`` object is kept in ``st.session_state`` so it survives
across reruns, and layout containers are declared in top-to-bottom
visual order up front, then filled after that run's actions have been
applied -- so the map and standings always reflect the latest tick.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

import simulator
import utils
from simulator import GameLogic, RaceState

st.set_page_config(page_title="Tactical Sailing Simulator", page_icon="⛵", layout="wide")

_COURSE_LABELS = {
    "windward_leeward": "Windward / Leeward (beat then run)",
    "triangle": "Triangle (beat, reach, reach)",
}


def _init_session_state() -> None:
    if "race_state" not in st.session_state:
        st.session_state.race_state = None


def _render_setup_sidebar() -> None:
    st.sidebar.header("⚙️ Race Setup")
    st.sidebar.slider(
        "Number of competitors",
        min_value=1,
        max_value=simulator.MAX_COMPETITORS,
        value=min(3, simulator.MAX_COMPETITORS),
        key="cfg_num_competitors",
        help=f"Capped at {simulator.MAX_COMPETITORS} so every boat keeps a distinct, colorblind-safe color.",
    )
    st.sidebar.slider("Wind speed (kts)", min_value=4, max_value=25, value=12, key="cfg_wind_speed")
    st.sidebar.slider(
        "Wind direction (° from)", min_value=0, max_value=359, value=0, key="cfg_wind_direction",
        help="Compass bearing the wind is blowing FROM.",
    )
    st.sidebar.selectbox(
        "Course",
        options=list(_COURSE_LABELS.keys()),
        format_func=lambda k: _COURSE_LABELS[k],
        key="cfg_course_type",
    )
    st.sidebar.slider(
        "First leg distance (m)", min_value=300, max_value=2000, value=800, step=100, key="cfg_race_distance",
    )
    seed_input = st.sidebar.number_input(
        "Random seed (0 = random)", min_value=0, value=0, step=1, key="cfg_seed",
    )

    if st.sidebar.button("🚩 Start Race", type="primary", width="stretch"):
        try:
            st.session_state.race_state = GameLogic.create_race(
                num_competitors=st.session_state.cfg_num_competitors,
                wind_speed_kts=float(st.session_state.cfg_wind_speed),
                wind_direction_deg=float(st.session_state.cfg_wind_direction),
                race_distance_m=float(st.session_state.cfg_race_distance),
                course_type=st.session_state.cfg_course_type,
                seed=int(seed_input) or None,
            )
            st.rerun()
        except ValueError as exc:
            st.sidebar.error(f"Couldn't start race: {exc}")

    if st.session_state.get("race_state") is not None:
        st.sidebar.divider()
        if st.sidebar.button("🔄 Reset / New Race", width="stretch"):
            st.session_state.race_state = None
            st.rerun()


def _render_welcome() -> None:
    st.title("⛵ Tactical Sailing Simulator")
    st.markdown(
        """
Configure your race in the sidebar and click **Start Race** to begin.

You'll skipper one boat in a fleet start. Every 10-second tick you can:

- **Tack / Jibe** across the wind
- Choose a **mark rounding** style (tight & risky vs. wide & safe)
- **Cover** a rival to blanket their wind
- Sail **conservative, balanced, or aggressive**

Watch the wind, the mark distance, and nearby rivals in the alert banner --
those are your tactical decision points.
        """
    )


def _render_controls(race_state: RaceState, context_before: dict) -> None:
    """Render the decision/control panel and apply any resulting actions."""
    boat = race_state.player_boat()

    st.subheader("🎮 Decisions")
    action_cols = st.columns([1, 1.3, 1.3, 1.6])

    with action_cols[0]:
        st.caption("Maneuver")
        tack_clicked = st.button(
            "⛵ Tack / Jibe",
            disabled=boat.status != simulator.BoatStatus.SAILING,
            width="stretch",
        )

    with action_cols[1]:
        st.caption("Next mark rounding")
        current_style = boat.pending_rounding_style or "standard"
        rounding_choice = st.radio(
            "Rounding style",
            options=list(simulator.ROUNDING_STYLES),
            index=list(simulator.ROUNDING_STYLES).index(current_style),
            key="ctl_rounding_style",
            label_visibility="collapsed",
            horizontal=True,
        )

    with action_cols[2]:
        st.caption("Cover a rival")
        nearby = context_before["nearby_competitors"]
        name_to_id = {name: bid for bid, name, _ in nearby}
        options = ["None"] + list(name_to_id.keys())
        current_name = next(
            (n for n, i in name_to_id.items() if i == boat.cover_target_id), "None"
        )
        cover_choice = st.selectbox(
            "Cover target",
            options=options,
            index=options.index(current_name) if current_name in options else 0,
            key="ctl_cover_target",
            label_visibility="collapsed",
        )

    with action_cols[3]:
        st.caption("Sailing style")
        style_choice = st.radio(
            "Sailing style",
            options=list(simulator.SAILING_STYLES),
            index=list(simulator.SAILING_STYLES).index(boat.sailing_style),
            key="ctl_sailing_style",
            label_visibility="collapsed",
            horizontal=True,
        )

    st.write("")
    advance_cols = st.columns([1, 1, 3])
    with advance_cols[0]:
        advance_clicked = st.button("▶ Advance 10s", type="primary", width="stretch")
    with advance_cols[1]:
        simulate_clicked = st.button("⏩ Simulate to Finish", width="stretch")

    # -- apply this run's actions, in a sensible order --
    GameLogic.set_rounding_style(race_state, rounding_choice)
    GameLogic.set_sailing_style(race_state, style_choice)
    GameLogic.set_cover_target(race_state, name_to_id.get(cover_choice))
    if tack_clicked:
        GameLogic.request_tack(race_state)
    if advance_clicked:
        GameLogic.tick(race_state)
    if simulate_clicked:
        GameLogic.run_autopilot(race_state)


def _render_banner(race_state: RaceState, context: dict) -> None:
    boat = race_state.player_boat()
    if boat.status == simulator.BoatStatus.FINISHED:
        return

    if context["approaching_mark"] and context["mark"] is not None:
        st.warning(
            f"⛳ Approaching **{context['mark'].name}** -- {context['distance_to_mark']:.0f}m out. "
            "Pick a rounding style below before you arrive."
        )
    if context["nearby_competitors"]:
        nearest = context["nearby_competitors"][0]
        st.info(f"👀 **{nearest[1]}** is only {nearest[2]:.0f}m away -- a covering opportunity (or a collision risk).")

    recent_shifts = [
        e for e in race_state.events
        if e.category == "wind_shift" and e.time >= race_state.elapsed_time - race_state.tick_seconds
    ]
    if recent_shifts:
        st.info(f"🌬️ {recent_shifts[-1].message}")

    if boat.status == simulator.BoatStatus.TACKING:
        st.caption(f"🔄 Tacking... {boat.tack_time_remaining:.0f}s remaining, sailing slow through the maneuver.")


def _render_telemetry(race_state: RaceState) -> None:
    boat = race_state.player_boat()
    cols = st.columns(5)
    cols[0].metric("Race Clock", utils.format_time(race_state.elapsed_time))
    cols[1].metric("Your Speed", f"{boat.speed:.1f} kts")
    cols[2].metric("Your Heading", f"{boat.heading:.0f}°", boat.tack_side)
    cols[3].metric("Wind", f"{race_state.wind.direction_deg:.0f}° @ {race_state.wind.speed_kts:.0f} kts")
    rank = next(
        (i + 1 for i, row in enumerate(utils.standings_dataframe(race_state).to_dict("records")) if row["Boat"] == "You"),
        None,
    )
    cols[4].metric("Position", f"{rank} / {len(race_state.boats)}" if rank else "-")

    st.dataframe(utils.standings_dataframe(race_state), hide_index=True, width="stretch")


def _render_results(race_state: RaceState) -> None:
    st.success("🏁 Race complete!")
    st.subheader("Final Standings")
    st.dataframe(utils.standings_dataframe(race_state), hide_index=True, width="stretch")

    st.subheader("Strategy Score")
    st.caption(
        "A heuristic 0-100 blend of finish time (relative to the fleet's fastest) and how "
        "clean the race was (fewer excess tacks and incidents)."
    )
    st.plotly_chart(utils.render_strategy_chart(race_state), width="stretch")

    with st.expander("📜 Decision Replay (full event log)"):
        st.dataframe(utils.decision_log_dataframe(race_state), hide_index=True, width="stretch")

    if st.button("🔁 Race Again", type="primary"):
        st.session_state.race_state = None
        st.rerun()


def main() -> None:
    _init_session_state()
    _render_setup_sidebar()

    race_state: Optional[RaceState] = st.session_state.race_state
    if race_state is None:
        _render_welcome()
        return

    st.title("⛵ Tactical Sailing Simulator")

    # Declare containers in top-to-bottom VISUAL order first; fill them
    # after this run's control actions have been applied (see below), so
    # everything on screen reflects the latest tick.
    map_slot = st.container()
    telemetry_slot = st.container()
    banner_slot = st.container()
    controls_slot = st.container()
    results_slot = st.container()

    context_before = GameLogic.get_decision_context(race_state)

    if race_state.status == "running":
        with controls_slot:
            _render_controls(race_state, context_before)
    else:
        with controls_slot:
            st.info("Race finished -- see full results below.")

    context_after = GameLogic.get_decision_context(race_state)

    with map_slot:
        st.plotly_chart(utils.render_race_map(race_state), width="stretch")
    with telemetry_slot:
        _render_telemetry(race_state)
    with banner_slot:
        _render_banner(race_state, context_after)
    if race_state.status == "finished":
        with results_slot:
            _render_results(race_state)


if __name__ == "__main__":
    main()
