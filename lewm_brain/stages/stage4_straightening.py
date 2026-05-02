"""Stage 4 — perceptual / neural straightening (Hénaff et al. 2021).

Inputs:
  - Stage 1 dataset: neural__{stim}.npz, unit_meta.csv
  - Stage 2 dataset: features__{stim}.npz
  - Pixel templates pulled fresh from the Allen warehouse via stimuli.py

Output:
  - straightening__{stim}.npz: per-step curvature series + windowed
    means in pixel / model / neural spaces, all on a shared 837-step
    grid (= n_movie_frames - clip_frames + 1) so the headline
    Δθ = θ_pixel - θ_repr is apples-to-apples.
  - sanity.png
  - config.json
  - Auto-publishes to Kaggle Dataset (cfg.stage4.dataset_id).

The "model" trajectory is the per-clip Stage 2 feature sequence. The
"neural" trajectory is per-frame population activity averaged over all
20 stimulus repeats — Hénaff 2021 averages across repeats before fitting
their AR(1) model so trial noise doesn't dominate the curvature.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import allen_data, kaggle_io, stimuli, straightening
from ..config import KAGGLE_WORKING, Config, write_artifact_manifest


def _deg(rad: float | np.ndarray) -> float | np.ndarray:
    return np.rad2deg(rad)


def run(
    cfg: Config,
    model_name: str = "vjepa2_vitl",
    stage1_root: Path = Path("/kaggle/input/lewm-brain-stage1"),
    stage2_root: Path = Path("/kaggle/input/lewm-brain-stage2"),
    out_root: Path | None = None,
) -> Path:
    out_root = Path(out_root) if out_root else (KAGGLE_WORKING / "stage4")
    stim = cfg.raw["stimulus"]["primary"]
    out_dir = out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    s4_cfg = cfg.raw.get("straightening") or {}
    win = int(s4_cfg.get("window_frames", 11))   # Hénaff 2021 default

    # 1. Load Stage 1 neural + Stage 2 features.
    print(f"[stage4] resolving neural__{stim}.npz (hint={stage1_root})")
    neural_path = kaggle_io.find_input_file(f"neural__{stim}.npz", stage1_root)
    print(f"[stage4]   found {neural_path}")
    print(f"[stage4] resolving features__{stim}.npz (hint={stage2_root})")
    feats_path = kaggle_io.find_input_file(f"features__{stim}.npz", stage2_root)
    print(f"[stage4]   found {feats_path}")

    neural = np.load(neural_path)
    feats = np.load(feats_path)
    responses = neural["responses"]                   # (R, T_total, U)
    unit_ids = neural["unit_ids"]
    features = feats["features"]                      # (n_clips, d)
    first_frame = int(feats["first_frame_with_feature"])
    R, T_total, U = responses.shape
    T_feat = features.shape[0]
    if T_total - first_frame < T_feat:
        raise RuntimeError(
            f"alignment: T_total={T_total}, first_frame={first_frame}, "
            f"T_feat={T_feat}"
        )

    # Y_mean: per-frame population vector, averaged across all repeats.
    Y_mean = (
        responses[:, first_frame : first_frame + T_feat, :]
        .astype(np.float32)
        .mean(axis=0)
    )                                                  # (T_feat, U)
    print(f"[stage4] features {features.shape}, Y_mean {Y_mean.shape}, "
          f"first_frame={first_frame}")

    # 2. Pixel movie. Allen warehouse, native grayscale, no resize. Fairer
    # comparison to "raw retinal input" than the 256² V-JEPA preprocess.
    print("[stage4] loading pixel template via Allen warehouse")
    manifest_path = KAGGLE_WORKING / "allen_cache" / "manifest.json"
    cache = allen_data.make_cache(manifest_path)
    pixels = stimuli.get_natural_movie_pixels(cache, stim)
    if pixels.ndim == 4:
        pixels = pixels.mean(axis=-1)                 # collapse RGB if any
    print(f"[stage4] pixels {pixels.shape} {pixels.dtype}")
    pix_aligned = pixels[first_frame : first_frame + T_feat]
    pix_flat = pix_aligned.reshape(T_feat, -1).astype(np.float32)

    # 3. Per-step curvature in each space (full 837-step trajectory).
    print(f"[stage4] curvature: pixel ({pix_flat.shape[1]} dims)")
    t0 = time.time()
    theta_pixel = straightening.per_step_curvature(pix_flat)
    t_pix_deg = float(np.nanmean(_deg(theta_pixel)))
    print(f"[stage4]   pixel  mean θ = {t_pix_deg:.2f}°  "
          f"({time.time() - t0:.1f}s)")

    theta_model = straightening.per_step_curvature(features.astype(np.float32))
    t_mod_deg = float(np.nanmean(_deg(theta_model)))
    print(f"[stage4]   model  mean θ = {t_mod_deg:.2f}°")

    theta_neural = straightening.per_step_curvature(Y_mean)
    t_neu_deg = float(np.nanmean(_deg(theta_neural)))
    print(f"[stage4]   neural mean θ = {t_neu_deg:.2f}°  "
          f"(pop vector, all {U} units)")

    # 3b. Per-area neural curvature. unit_meta is indexed by unit_id; the
    # column ordering in Y_mean follows `unit_ids` from the Stage 1 npz.
    per_area_theta_deg: dict[str, float] = {}
    unit_meta = None
    try:
        meta_path = kaggle_io.find_input_file("unit_meta.csv", stage1_root)
        unit_meta = pd.read_csv(meta_path, index_col=0).loc[unit_ids]
        for area, sub in unit_meta.groupby("ecephys_structure_acronym"):
            cols = np.where(np.isin(unit_ids, sub.index.to_numpy()))[0]
            if cols.size < 2:
                continue
            theta_a = straightening.per_step_curvature(Y_mean[:, cols])
            per_area_theta_deg[str(area)] = float(np.nanmean(_deg(theta_a)))
        if per_area_theta_deg:
            line = ", ".join(f"{a}={v:.1f}°"
                             for a, v in per_area_theta_deg.items())
            print(f"[stage4] per-area neural θ: {line}")
    except FileNotFoundError:
        print("[stage4] unit_meta.csv not found; skipping per-area neural")

    # 4. Windowed (Hénaff 2021 style: ~11-frame natural-movie clips).
    win_pixel = straightening.windowed_mean_curvature(pix_flat, win=win)
    win_model = straightening.windowed_mean_curvature(
        features.astype(np.float32), win=win,
    )
    win_neural = straightening.windowed_mean_curvature(Y_mean, win=win)
    print(f"[stage4] windowed (win={win}): "
          f"pixel mean {_deg(np.nanmean(win_pixel)):.1f}°, "
          f"model {_deg(np.nanmean(win_model)):.1f}°, "
          f"neural {_deg(np.nanmean(win_neural)):.1f}°")

    # 5. Sanity plot. Two panels: per-window distribution + a global bar.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(_deg(win_pixel), bins=30, alpha=0.5,
                 label="pixel", color="C0")
    axes[0].hist(_deg(win_model), bins=30, alpha=0.5,
                 label=f"model ({model_name})", color="C1")
    axes[0].hist(_deg(win_neural), bins=30, alpha=0.5,
                 label="neural (pop.)", color="C2")
    axes[0].set_xlabel(f"mean θ within {win}-frame window (°)")
    axes[0].set_ylabel("# windows")
    axes[0].set_title(f"{stim}: per-window curvature")
    axes[0].legend()

    bars = ["pixel", f"model\n({model_name})", "neural\n(pop)"]
    vals = [t_pix_deg, t_mod_deg, t_neu_deg]
    axes[1].bar(bars, vals, color=["C0", "C1", "C2"])
    axes[1].set_ylabel("mean θ (°)")
    axes[1].set_title(
        f"global trajectory curvature\n"
        f"Δ(pix-model)={t_pix_deg - t_mod_deg:+.1f}°, "
        f"Δ(pix-neural)={t_pix_deg - t_neu_deg:+.1f}°"
    )
    axes[1].axhline(90, color="k", lw=0.5, ls="--")
    fig.tight_layout()
    fig.savefig(out_dir / "sanity.png", dpi=110)
    plt.close(fig)

    # 6. Save arrays.
    np.savez_compressed(
        out_dir / f"straightening__{stim}.npz",
        theta_pixel=theta_pixel,
        theta_model=theta_model,
        theta_neural=theta_neural,
        win_pixel=win_pixel,
        win_model=win_model,
        win_neural=win_neural,
        unit_ids=unit_ids,
        first_frame=first_frame,
        window_frames=win,
    )

    write_artifact_manifest(
        out_dir, cfg,
        extra={
            "stage": "stage4_straightening",
            "model_name": model_name,
            "stimulus": stim,
            "n_timesteps": int(T_feat),
            "window_frames": int(win),
            "mean_theta_deg": {
                "pixel": t_pix_deg,
                "model": t_mod_deg,
                "neural": t_neu_deg,
            },
            "delta_pixel_minus_model_deg": float(t_pix_deg - t_mod_deg),
            "delta_pixel_minus_neural_deg": float(t_pix_deg - t_neu_deg),
            "per_area_neural_theta_deg": per_area_theta_deg,
        },
    )
    print(f"[stage4] wrote {out_dir}")
    print(f"[stage4] Δ_pix-model  = {t_pix_deg - t_mod_deg:+.1f}°  "
          f"(positive = model trajectory is straighter than pixels)")
    print(f"[stage4] Δ_pix-neural = {t_pix_deg - t_neu_deg:+.1f}°  "
          f"(positive = cortical population is straighter than pixels — "
          f"Hénaff 2021 prediction)")

    # 7. Auto-publish.
    s4 = cfg.raw.get("stage4") or {}
    ds_id = s4.get("dataset_id")
    if ds_id:
        ds_title = s4.get("dataset_title", f"lewm-brain stage4 ({model_name})")
        try:
            kaggle_io.publish_to_kaggle_dataset(out_dir, ds_id, ds_title)
        except Exception as exc:
            print(f"[stage4] WARN auto-publish raised: {exc!r}")

    return out_dir
