"""Allen Brain Observatory Neuropixels — session selection and neural extraction.

Session selection still goes through `EcephysProjectCache.get_units()` (works,
no pynwb). For the per-session NWB file we bypass AllenSDK's pynwb-based
session loader (which is broken on the current Kaggle pynwb/hdmf versions —
`NWBFile` has an abstract `external_resources` method that AllenSDK never
implemented in its subclass) and read the HDF5 file directly with h5py.

Default unit filter follows the AllenSDK quality-metrics tutorial
(`isi_violations < 0.5`, `amplitude_cutoff < 0.1`, `presence_ratio > 0.9`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .config import Config

# Six visual cortical areas targeted by the Allen Visual Coding probes.
VISUAL_CORTEX_AREAS = ("VISp", "VISl", "VISal", "VISpm", "VISam", "VISrl")


# ---------------------------------------------------------------------------
# AllenSDK monkey-patch: skip the test=True validation that blows up because
# of the pynwb/hdmf abstract-class mismatch. We don't need any EcephysSession
# methods — only the local file path of the downloaded NWB. With the patch,
# `cache.get_session_data(session_id)` returns an instance whose `api.path`
# points at the cached .nwb file, and AllenSDK no longer deletes it on a
# failed read.
# ---------------------------------------------------------------------------
def _install_ecephys_session_patch() -> None:
    from allensdk.brain_observatory.ecephys import ecephys_session as _mod
    cls = _mod.EcephysSession
    if getattr(cls, "_lewm_patched", False):
        return
    _orig_init = cls.__init__

    def _patched_init(self, api, test=False, **kwargs):
        # Force test=False; we read the NWB ourselves via h5py.
        return _orig_init(self, api, test=False, **kwargs)

    cls.__init__ = _patched_init
    cls._lewm_patched = True


def make_cache(manifest_path: Path):
    """Construct an EcephysProjectCache backed by AllenSDK's S3 warehouse."""
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
        EcephysProjectCache,
    )
    _install_ecephys_session_patch()
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
    """Apply the deterministic session-pick rule using the warehouse units
    table (which works — pynwb is not involved).
    """
    sp = cfg.raw["session_pick"]
    higher = list(sp["any_higher_areas"])
    visual = list(sp["required_areas"]) + higher

    sessions = cache.get_session_table()
    units = cache.get_units()
    units = _passes_default_filter(units, cfg)

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
            f"any_higher={higher}."
        )
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


def get_session_nwb_path(cache, session_id: int) -> Path:
    """Make sure the session's NWB is on disk and return its path. With our
    monkey-patch, this download succeeds without the test=True validation.
    """
    session = cache.get_session_data(session_id)
    return Path(session.api.path)


# ---------------------------------------------------------------------------
# Direct h5py readers — replace the broken pynwb/EcephysSession data path.
# HDF5 paths verified against the GroupBuilder dump from a real Allen NWB.
# ---------------------------------------------------------------------------
def _decode_strs(arr: np.ndarray) -> np.ndarray:
    if arr.size and isinstance(arr.flat[0], (bytes, np.bytes_)):
        return np.array([x.decode() if isinstance(x, (bytes, np.bytes_)) else x
                         for x in arr.tolist()])
    return arr


def good_unit_ids_in_visual_cortex(
    nwb_path: Path, cfg: Config,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Read /units, apply default quality filter, restrict to visual cortex.
    Returns (unit_ids, unit_metadata_df indexed by unit_id).
    """
    with h5py.File(nwb_path, "r") as f:
        u = f["units"]
        units = pd.DataFrame({
            "id": u["id"][:],
            "isi_violations": u["isi_violations"][:],
            "amplitude_cutoff": u["amplitude_cutoff"][:],
            "presence_ratio": u["presence_ratio"][:],
            "peak_channel_id": u["peak_channel_id"][:],
        }).set_index("id")
        e = f["general/extracellular_ephys/electrodes"]
        elec_ids = e["id"][:]
        locations = _decode_strs(e["location"][:])

    elec_to_area = pd.Series(locations, index=elec_ids)
    units["ecephys_structure_acronym"] = units["peak_channel_id"].map(elec_to_area)

    units = _passes_default_filter(units, cfg)
    units = units[units["ecephys_structure_acronym"].isin(VISUAL_CORTEX_AREAS)]
    return units.index.to_numpy(), units


def _load_movie_presentations(nwb_path: Path, stim_name: str) -> pd.DataFrame:
    """Read /intervals/{stim}_presentations and derive a `repeat` column from
    the per-frame structure (each `frame == 0` marks a new repeat).
    """
    grp_name = f"intervals/{stim_name}_presentations"
    with h5py.File(nwb_path, "r") as f:
        if grp_name not in f:
            raise FileNotFoundError(f"NWB has no {grp_name!r}")
        g = f[grp_name]
        df = pd.DataFrame({
            "start_time": g["start_time"][:],
            "stop_time": g["stop_time"][:],
            "frame": g["frame"][:].astype(int),
            "stimulus_block": g["stimulus_block"][:].astype(int),
        })
    df = df.sort_values("start_time").reset_index(drop=True)
    # Each `frame == 0` row begins a new repeat; cumulative count gives a
    # 0-based repeat index across the whole session (across stimulus_block
    # boundaries too, which is what we want — 20 repeats total in BO 1.1
    # for natural_movie_one).
    df["repeat"] = (df["frame"] == 0).cumsum() - 1
    return df


def bin_responses_per_frame(
    nwb_path: Path,
    stimulus_name: str,
    unit_ids: np.ndarray,
    cfg: Config,  # kept for signature parity
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return spike counts shaped (n_repeats, n_frames, n_units) plus the
    sorted presentation table. One bin per presentation = one movie frame.
    """
    pres = _load_movie_presentations(nwb_path, stimulus_name)

    with h5py.File(nwb_path, "r") as f:
        all_unit_ids = f["units/id"][:]
        spike_index = f["units/spike_times_index"][:]
        # Read only the slice of spike_times we need (cheaper than full load
        # for sessions where many units exist outside our visual-cortex set).
        spike_times = f["units/spike_times"]
        # Slot lookup unit_id -> position in spike_times_index.
        pos_by_id = pd.Series(np.arange(len(all_unit_ids)), index=all_unit_ids)
        if not pos_by_id.index.is_unique:
            raise RuntimeError("Duplicate unit ids in NWB /units/id.")

        starts = pres["start_time"].to_numpy()
        stops = pres["stop_time"].to_numpy()
        n_pres = len(starts)
        n_units = len(unit_ids)
        counts = np.zeros((n_pres, n_units), dtype=np.int32)

        for u_idx, uid in enumerate(unit_ids):
            pos = int(pos_by_id.loc[uid])
            sl_start = int(spike_index[pos - 1]) if pos > 0 else 0
            sl_stop = int(spike_index[pos])
            if sl_stop <= sl_start:
                continue
            unit_spikes = spike_times[sl_start:sl_stop]
            left = np.searchsorted(unit_spikes, starts, side="left")
            right = np.searchsorted(unit_spikes, stops, side="left")
            counts[:, u_idx] = right - left

    n_repeats = int(pres["repeat"].nunique())
    per_repeat = pres.groupby("repeat").size()
    if per_repeat.nunique() != 1:
        raise RuntimeError(
            f"Uneven frames per repeat for {stimulus_name!r}: "
            f"{per_repeat.to_dict()}."
        )
    n_frames = int(per_repeat.iloc[0])
    if counts.shape[0] != n_repeats * n_frames:
        raise RuntimeError(
            f"Presentation count mismatch: {counts.shape[0]} rows vs "
            f"n_repeats={n_repeats} × n_frames={n_frames}."
        )
    responses = counts.reshape(n_repeats, n_frames, n_units)
    return responses, pres


def running_speed_per_frame(
    nwb_path: Path,
    pres_sorted: pd.DataFrame,
    cfg: Config,  # kept for signature parity
) -> np.ndarray:
    """Resample running speed (cm/s) to per-frame midpoints, return shape
    (n_repeats, n_frames).
    """
    with h5py.File(nwb_path, "r") as f:
        rs = f["processing/running/running_speed"]
        rs_t = rs["timestamps"][:]
        rs_v = rs["data"][:]

    midpoints = (
        pres_sorted["start_time"].to_numpy()
        + pres_sorted["stop_time"].to_numpy()
    ) / 2.0
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
