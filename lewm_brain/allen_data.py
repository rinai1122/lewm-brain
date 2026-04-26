"""Allen Brain Observatory Neuropixels — session selection and neural extraction.

API references and unverified assumptions are tagged so the first Kaggle
run can confirm them. Default unit filter follows the AllenSDK quality-
metrics tutorial (`isi_violations < 0.5`, `amplitude_cutoff < 0.1`,
`presence_ratio > 0.9`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config

# Six visual cortical areas targeted by the Allen Visual Coding probes.
# brain-map.org/circuits-behavior/visual-coding-neuropixels.
VISUAL_CORTEX_AREAS = ("VISp", "VISl", "VISal", "VISpm", "VISam", "VISrl")


def make_cache(manifest_path: Path):
    """Construct an EcephysProjectCache backed by AllenSDK's S3 warehouse.

    Imports are deferred so this module stays importable on the laptop
    (no allensdk locally) and only blows up when actually used on Kaggle.
    """
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
        EcephysProjectCache,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return EcephysProjectCache.from_warehouse(manifest=str(manifest_path))


def _passes_default_filter(units: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    f = cfg.raw["unit_filter"]
    return units[
        (units["isi_violations"] < f["isi_violations_max"])
        & (units["amplitude_cutoff"] < f["amplitude_cutoff_max"])
        & (units["presence_ratio"] > f["presence_ratio_min"])
    ]


def pick_session_id(cache, cfg: Config) -> tuple[int, dict[str, Any]]:
    """Apply the deterministic session-pick rule.

    Returns (session_id, info) where info describes why this session won.
    Implements:
      session_type == cfg.session_pick.session_type AND
      session has VISp AND
      session has ≥ 1 of cfg.session_pick.any_higher_areas AND
      argmax over (count of good units in those areas).
    Tie-break by lowest session_id.
    """
    sp = cfg.raw["session_pick"]
    higher = list(sp["any_higher_areas"])
    visual = list(sp["required_areas"]) + higher

    sessions = cache.get_session_table()
    units = cache.get_units()  # one row per unit, all sessions
    units = _passes_default_filter(units, cfg)

    # The unit table column for cortical area is `ecephys_structure_acronym`.
    units_vis = units[units["ecephys_structure_acronym"].isin(visual)]
    grp = units_vis.groupby("ecephys_session_id")["ecephys_structure_acronym"]
    summary = pd.DataFrame({
        "good_units_visual": grp.size(),
        "has_VISp": grp.apply(lambda s: "VISp" in set(s)),
        "n_higher_areas": grp.apply(lambda s: len(set(s) & set(higher))),
    })
    eligible = summary[(summary["has_VISp"]) & (summary["n_higher_areas"] >= 1)]
    eligible = eligible.merge(
        sessions[["session_type"]], left_index=True, right_index=True,
    )
    eligible = eligible[eligible["session_type"] == sp["session_type"]]

    if eligible.empty:
        raise RuntimeError(
            "No sessions satisfy the session-pick rule. "
            f"session_type={sp['session_type']!r}, required={sp['required_areas']}, "
            f"any_higher={higher}. Check the unit-filter thresholds."
        )

    # Stable sort of (good_units_visual desc) on a session-id-ASC indexed
    # frame -> tie-break is lowest session_id.
    eligible = eligible.sort_index().sort_values(
        "good_units_visual", ascending=False, kind="mergesort",
    )
    session_id = int(eligible.index[0])
    info = {
        "good_units_visual": int(eligible.iloc[0]["good_units_visual"]),
        "n_higher_areas": int(eligible.iloc[0]["n_higher_areas"]),
        "n_eligible_sessions": int(len(eligible)),
    }
    return session_id, info


def load_session(cache, session_id: int):
    """Pull one session's NWB file (downloads to cache on first call)."""
    return cache.get_session_data(session_id)


def good_unit_ids_in_visual_cortex(session, cfg: Config) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (unit_ids, unit_metadata_df) for all visual-cortex units that
    pass the default filter for this session.
    """
    units = session.units  # DataFrame indexed by unit_id
    units = _passes_default_filter(units, cfg)
    units = units[units["ecephys_structure_acronym"].isin(VISUAL_CORTEX_AREAS)]
    return units.index.to_numpy(), units


def bin_responses_per_frame(
    session,
    stimulus_name: str,
    unit_ids: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return spike counts shaped (n_repeats, n_frames, n_units) plus the
    presentation table.

    Assumption (to verify on first Kaggle run): each row of the natural-
    movie stimulus table is one frame, with `repeat` and `frame` columns
    and `stop_time - start_time ≈ 1/30 s`. AllenSDK's worked examples
    indicate this; if the grain turns out to be one row per repeat
    instead, the `pivot` step below will fail loudly and we adapt.
    """
    fps = cfg.raw["stimulus"]["frame_rate_hz"]
    pres = session.get_stimulus_table(stimulus_name)
    if "repeat" not in pres.columns or "frame" not in pres.columns:
        raise RuntimeError(
            f"Stimulus table for {stimulus_name!r} missing 'repeat' or "
            f"'frame' columns — got {list(pres.columns)}. Update the "
            "binning logic for this stimulus grain."
        )

    pres_sorted = pres.sort_values(["repeat", "frame"])
    bin_edges = np.array([0.0, 1.0 / fps])
    counts = session.presentationwise_spike_counts(
        bin_edges=bin_edges,
        stimulus_presentation_ids=pres_sorted.index.to_numpy(),
        unit_ids=unit_ids,
    )
    # Force the xarray to the (repeat, frame)-sorted order so reshape is
    # unambiguous regardless of how AllenSDK orders its return value.
    counts = counts.sel(stimulus_presentation_id=pres_sorted.index.to_numpy())
    arr_sorted = counts.values.squeeze(axis=1)  # (n_pres, n_units)

    n_repeats = int(pres_sorted["repeat"].nunique())
    per_repeat = pres_sorted.groupby("repeat").size()
    if per_repeat.nunique() != 1:
        raise RuntimeError(
            f"Uneven frames per repeat for {stimulus_name!r}: {per_repeat.to_dict()}. "
            "bin_responses_per_frame assumes uniform frame counts."
        )
    n_frames = int(per_repeat.iloc[0])
    if arr_sorted.shape[0] != n_repeats * n_frames:
        raise RuntimeError(
            f"Presentation count mismatch: {arr_sorted.shape[0]} rows but "
            f"n_repeats={n_repeats} × n_frames={n_frames}."
        )
    responses = arr_sorted.reshape(n_repeats, n_frames, len(unit_ids))
    return responses, pres_sorted


def running_speed_per_frame(
    session,
    pres_sorted: pd.DataFrame,
    cfg: Config,
) -> np.ndarray:
    """Resample running speed onto the per-frame timing of the stimulus.

    Assumption: `session.running_speed` is a DataFrame with start_time /
    end_time / velocity columns sampled finer than 30 Hz; we take the
    last sample whose start_time ≤ frame midpoint.
    """
    rs = session.running_speed  # AllenSDK DataFrame
    midpoints = (
        pres_sorted["start_time"].to_numpy()
        + pres_sorted["stop_time"].to_numpy()
    ) / 2.0
    rs_t = rs["start_time"].to_numpy()
    rs_v = rs["velocity"].to_numpy()
    idx = np.searchsorted(rs_t, midpoints, side="right") - 1
    idx = np.clip(idx, 0, len(rs_v) - 1)
    velocities = rs_v[idx]
    n_repeats = int(pres_sorted["repeat"].nunique())
    n_frames = len(velocities) // n_repeats
    return velocities.reshape(n_repeats, n_frames)


def per_area_unit_counts(unit_meta: pd.DataFrame) -> dict[str, int]:
    return (
        unit_meta["ecephys_structure_acronym"]
        .value_counts()
        .reindex(VISUAL_CORTEX_AREAS, fill_value=0)
        .astype(int)
        .to_dict()
    )
