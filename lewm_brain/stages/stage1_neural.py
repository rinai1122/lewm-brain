"""Stage 1 — neural data prep.

Picks one Allen Visual Coding Neuropixels session by the deterministic
rule in the config, extracts per-frame binned spike counts and running
speed for each natural-movie stimulus, and writes:

    {out_dir}/neural__{stimulus}.npz   # responses + metadata
    {out_dir}/sanity.png               # eyeball plot
    {out_dir}/config.json              # resolved config + commit hash

`out_dir` defaults to `/kaggle/working/stage1/{session_id}/`.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import allen_data, kaggle_io
from ..config import KAGGLE_WORKING, Config, write_artifact_manifest


def _save_arrays(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _sanity_plot(
    out_path: Path,
    stim_name: str,
    responses: np.ndarray,
    unit_meta,
    per_area: dict[str, int],
) -> None:
    """One PNG: per-area unit counts (left) + (units × frames) heatmap
    for repeat 0 (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1, 3]})
    areas = list(per_area.keys())
    counts = [per_area[a] for a in areas]
    axes[0].bar(areas, counts)
    axes[0].set_title("Good units per visual-cortex area")
    axes[0].set_ylabel("count")
    axes[0].tick_params(axis="x", rotation=30)

    # Group units visibly by area in the heatmap.
    sorted_meta = unit_meta.sort_values("ecephys_structure_acronym")
    pos = {uid: i for i, uid in enumerate(unit_meta.index)}
    perm = np.array([pos[u] for u in sorted_meta.index])
    rep0 = responses[0][:, perm].T  # (n_units, n_frames)

    im = axes[1].imshow(
        rep0, aspect="auto", interpolation="nearest", origin="lower",
    )
    axes[1].set_title(f"{stim_name} — repeat 0 (units × frames spike count)")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("unit (sorted by area)")
    fig.colorbar(im, ax=axes[1], label="spikes / frame")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run(cfg: Config, out_root: Path | None = None) -> Path:
    out_root = Path(out_root) if out_root else (KAGGLE_WORKING / "stage1")

    # 1. AllenSDK cache. Kaggle scratch is /kaggle/working — use it for the
    # manifest; sessions land alongside the manifest.
    manifest_path = KAGGLE_WORKING / "allen_cache" / "manifest.json"
    cache = allen_data.make_cache(manifest_path)

    # 2. Session pick.
    session_id, pick_info = allen_data.pick_session_id(cache, cfg)
    cfg.set("session_pick", "session_id", value=session_id)
    print(f"[stage1] picked session_id={session_id} ({pick_info})")

    out_dir = out_root / str(session_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Download session NWB (via our monkey-patched cache so AllenSDK
    # doesn't auto-delete on read-failure), then read it directly with h5py.
    t0 = time.time()
    nwb_path = allen_data.get_session_nwb_path(cache, session_id)
    print(f"[stage1] NWB ready in {time.time() - t0:.1f}s at {nwb_path}")

    unit_ids, unit_meta = allen_data.good_unit_ids_in_visual_cortex(nwb_path, cfg)
    print(f"[stage1] {len(unit_ids)} good visual-cortex units")
    per_area = allen_data.per_area_unit_counts(unit_meta)
    print(f"[stage1] per-area counts: {per_area}")

    unit_meta_csv = out_dir / "unit_meta.csv"
    unit_meta.to_csv(unit_meta_csv)

    # 4. Per-stimulus binning + running speed (h5py-direct).
    stims = [cfg.raw["stimulus"]["primary"]] + list(cfg.raw["stimulus"]["also"])
    summary = {}
    for stim in stims:
        try:
            responses, pres_sorted = allen_data.bin_responses_per_frame(
                nwb_path, stim, unit_ids, cfg,
            )
            running = allen_data.running_speed_per_frame(nwb_path, pres_sorted, cfg)
        except Exception as exc:
            # FC sessions don't have natural_movie_three; log and skip.
            print(f"[stage1] skipping {stim}: {exc}")
            continue

        _save_arrays(
            out_dir / f"neural__{stim}.npz",
            responses=responses,
            running_speed=running,
            unit_ids=unit_ids,
            start_times=pres_sorted["start_time"].to_numpy(),
            stop_times=pres_sorted["stop_time"].to_numpy(),
            repeats=pres_sorted["repeat"].to_numpy(),
            frames=pres_sorted["frame"].to_numpy(),
        )
        summary[stim] = {
            "responses_shape": list(responses.shape),
            "running_speed_shape": list(running.shape),
        }
        print(f"[stage1] {stim}: responses {responses.shape}, "
              f"running {running.shape}")

        if stim == cfg.raw["stimulus"]["primary"]:
            _sanity_plot(
                out_dir / "sanity.png",
                stim, responses, unit_meta, per_area,
            )

        # Snapshot to Kaggle Dataset right after each stim writes — so a
        # later kernel crash can't lose what we already extracted.
        s1 = cfg.raw.get("stage1") or {}
        ds_id = s1.get("dataset_id")
        if ds_id:
            ds_title = s1.get(
                "dataset_title",
                f"lewm-brain stage 1 (session {session_id})",
            )
            try:
                kaggle_io.publish_to_kaggle_dataset(out_dir, ds_id, ds_title)
            except Exception as exc:
                print(f"[stage1] WARN auto-publish raised: {exc!r}")

    write_artifact_manifest(
        out_dir, cfg,
        extra={
            "stage": "stage1_neural",
            "session_id": session_id,
            "session_pick_info": pick_info,
            "n_good_units": int(len(unit_ids)),
            "per_area_counts": per_area,
            "stimuli_summary": summary,
        },
    )
    print(f"[stage1] wrote {out_dir}")
    return out_dir
