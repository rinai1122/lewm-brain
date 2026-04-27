"""Stage 3 — ridge encoding model.

Inputs:
  - Stage 1 dataset: neural__{stim}.npz, unit_meta.csv  (any nested layout)
  - Stage 2 dataset: features__{stim}.npz                (any nested layout)

Output:
  - encoding__{stim}.npz: per-unit Pearson r on the held-out repeat,
    per-unit alpha, plus split-half reliability noise ceiling.
  - sanity.png: r vs reliability scatter, per-area mean bar.
  - config.json
  - Auto-publishes to Kaggle Dataset (cfg.stage3.dataset_id).
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import encoding, kaggle_io
from ..config import KAGGLE_WORKING, Config, write_artifact_manifest


def _find(root: Path, filename: str) -> Path:
    for dirpath, _, files in __import__("os").walk(root):
        if filename in files:
            return Path(dirpath) / filename
    raise FileNotFoundError(f"{filename!r} not found anywhere under {root}")


def _per_area_mean(values: np.ndarray, areas: pd.Series) -> dict[str, float]:
    """Mean of `values` indexed by area (Pandas Series indexed by unit_id)."""
    df = pd.DataFrame({"v": values, "area": areas.to_numpy()})
    return df.groupby("area")["v"].mean().to_dict()


def run(
    cfg: Config,
    model_name: str = "vjepa2_vitl",
    stage1_root: Path = Path("/kaggle/input/lewm-brain-stage1"),
    stage2_root: Path = Path("/kaggle/input/lewm-brain-stage2"),
    out_root: Path | None = None,
) -> Path:
    out_root = Path(out_root) if out_root else (KAGGLE_WORKING / "stage3")
    stim = cfg.raw["stimulus"]["primary"]
    out_dir = out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load inputs.
    print(f"[stage3] looking under {stage1_root} for neural__{stim}.npz")
    neural_path = _find(stage1_root, f"neural__{stim}.npz")
    print(f"[stage3]   found {neural_path}")
    print(f"[stage3] looking under {stage2_root} for features__{stim}.npz")
    feats_path = _find(stage2_root, f"features__{stim}.npz")
    print(f"[stage3]   found {feats_path}")

    neural = np.load(neural_path)
    feats = np.load(feats_path)
    responses = neural["responses"]      # (n_repeats, n_total_frames, n_units)
    unit_ids = neural["unit_ids"]         # (n_units,)
    features = feats["features"]          # (n_clips, d_model)
    first_frame = int(feats["first_frame_with_feature"])  # = clip_frames - 1
    print(f"[stage3] responses {responses.shape}, features {features.shape}, "
          f"first_frame_with_feature={first_frame}")

    # 2. Align time axes. Drop the first `first_frame` frames in responses
    # (no feature available for them).
    R, T_total, U = responses.shape
    T_feat = features.shape[0]
    responses_aligned = responses[:, first_frame:first_frame + T_feat, :]
    if responses_aligned.shape[1] != T_feat:
        raise RuntimeError(
            f"alignment mismatch: responses[{first_frame}:{first_frame + T_feat}] "
            f"-> {responses_aligned.shape}, but features have T={T_feat}."
        )

    # 3. Train/test split: leave the LAST repeat out as held-out test.
    n_train = R - 1
    Y_train = responses_aligned[:n_train].astype(np.float32).mean(axis=0)
    Y_test = responses_aligned[-1].astype(np.float32)
    X = features.astype(np.float32)
    print(f"[stage3] X {X.shape}, Y_train {Y_train.shape} "
          f"(avg over {n_train} repeats), Y_test {Y_test.shape}")

    # 4. Fit ridge per neuron, score on held-out repeat.
    alpha_grid = list(cfg.raw.get("encoder", {}).get(
        "alpha_grid", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]))
    t0 = time.time()
    result = encoding.fit_and_score_ridge(X, Y_train, Y_test, alpha_grid)
    print(f"[stage3] ridge fit in {time.time() - t0:.1f}s; "
          f"per-unit r mean={np.nanmean(result['r']):.3f} "
          f"median={np.nanmedian(result['r']):.3f}")

    # 5. Noise ceiling (split-half reliability across all repeats).
    rel = encoding.split_half_reliability(responses_aligned)
    print(f"[stage3] split-half reliability mean={np.nanmean(rel):.3f}")

    # 6. Per-area summary if unit_meta available.
    per_area_r = {}
    per_area_rel = {}
    try:
        meta_path = _find(stage1_root, "unit_meta.csv")
        unit_meta = pd.read_csv(meta_path, index_col=0)
        unit_meta = unit_meta.loc[unit_ids]
        per_area_r = _per_area_mean(result["r"],
                                    unit_meta["ecephys_structure_acronym"])
        per_area_rel = _per_area_mean(rel,
                                      unit_meta["ecephys_structure_acronym"])
        print(f"[stage3] per-area mean r: "
              + ", ".join(f"{a}={v:.3f}" for a, v in per_area_r.items()))
    except FileNotFoundError:
        print("[stage3] unit_meta.csv not found; skipping per-area summary")
        unit_meta = None

    # 7. Sanity plot.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(rel, result["r"], s=6, alpha=0.5)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[0].set_xlabel("split-half reliability (noise ceiling)")
    axes[0].set_ylabel("encoding r (held-out repeat)")
    axes[0].set_title(f"{model_name} on {stim} — n_units={len(result['r'])}")
    axes[0].set_xlim(-0.2, 1)
    axes[0].set_ylim(-0.2, 1)

    if per_area_r:
        areas = list(per_area_r.keys())
        rs = [per_area_r[a] for a in areas]
        axes[1].bar(areas, rs)
        axes[1].set_title("Mean r per area")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "sanity.png", dpi=110)
    plt.close(fig)

    # 8. Save outputs.
    np.savez_compressed(
        out_dir / f"encoding__{stim}.npz",
        r=result["r"],
        alpha=result["alpha"],
        reliability=rel,
        unit_ids=unit_ids,
        Y_pred=result["Y_pred"],
    )

    write_artifact_manifest(
        out_dir, cfg,
        extra={
            "stage": "stage3_encoder",
            "model_name": model_name,
            "stimulus": stim,
            "n_units": int(len(result["r"])),
            "r_mean": float(np.nanmean(result["r"])),
            "r_median": float(np.nanmedian(result["r"])),
            "reliability_mean": float(np.nanmean(rel)),
            "per_area_r": per_area_r,
            "per_area_rel": per_area_rel,
            "alpha_grid": alpha_grid,
        },
    )
    print(f"[stage3] wrote {out_dir}")

    # 9. Auto-publish.
    s3 = cfg.raw.get("stage3") or {}
    ds_id = s3.get("dataset_id")
    if ds_id:
        ds_title = s3.get("dataset_title", f"lewm-brain stage3 ({model_name})")
        try:
            kaggle_io.publish_to_kaggle_dataset(out_dir, ds_id, ds_title)
        except Exception as exc:
            print(f"[stage3] WARN auto-publish raised: {exc!r}")

    return out_dir
