"""Stage 2 — backbone feature extraction over natural-movie clips.

Defaults to `facebook/vjepa2-vitl-fpc64-256`, sliding stride 1.
Outputs `features__{stim}.npz` with shape (n_clips, hidden_size) per
stimulus, plus a config.json that ties this run to the model id, init
mode, and clip-window choice.

Per-frame assignment for downstream stages: clip k's feature is assigned
to movie frame `k + clip_frames - 1` (i.e. the last frame in the clip).
The first `clip_frames - 1` frames per movie repeat have no feature.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .. import allen_data, features, stimuli
from ..config import KAGGLE_WORKING, Config, write_artifact_manifest


def run(
    cfg: Config,
    model_index: int = 0,
    out_root: Path | None = None,
) -> Path:
    out_root = Path(out_root) if out_root else (KAGGLE_WORKING / "stage2")

    model_cfg = cfg.raw["models"][model_index]
    model_name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]
    clip_frames = int(model_cfg["clip_frames"])
    init = model_cfg.get("init", "pretrained")
    seed = int(cfg.raw.get("seed", 0))

    out_dir = out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model + processor.
    print(f"[stage2] loading {hf_id} (init={init})")
    import torch
    model, processor = features.load_vjepa2(
        hf_id, init=init, seed=seed, dtype="float16",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage2] device={device}, model dtype={next(model.parameters()).dtype}")

    # 2. AllenSDK cache for the pixel templates.
    manifest_path = KAGGLE_WORKING / "allen_cache" / "manifest.json"
    cache = allen_data.make_cache(manifest_path)

    # 3. For each stimulus: load pixels -> sliding-window features.
    stims = [cfg.raw["stimulus"]["primary"]] + list(cfg.raw["stimulus"]["also"])
    summary = {}
    for stim in stims:
        print(f"[stage2] {stim}: loading pixel template")
        pixels = stimuli.get_natural_movie_pixels(cache, stim)
        print(f"[stage2] {stim} pixels: shape={pixels.shape}, dtype={pixels.dtype}")

        t0 = time.time()
        feats = features.vjepa2_extract_features(
            model, processor, pixels,
            clip_frames=clip_frames,
            stride=1,
            batch_size=int(cfg.raw.get("stage2_batch_size", 1)),
            device=device,
            pool="mean",
        )
        elapsed = time.time() - t0
        print(f"[stage2] {stim}: features {feats.shape} in {elapsed:.1f}s "
              f"({elapsed / max(1, feats.shape[0]) * 1000:.1f} ms/clip)")

        np.savez_compressed(
            out_dir / f"features__{stim}.npz",
            features=feats,
            n_movie_frames=int(pixels.shape[0]),
            clip_frames=clip_frames,
            stride=1,
            first_frame_with_feature=clip_frames - 1,
        )
        summary[stim] = {
            "features_shape": list(feats.shape),
            "elapsed_s": elapsed,
            "n_movie_frames": int(pixels.shape[0]),
        }

    write_artifact_manifest(
        out_dir, cfg,
        extra={
            "stage": "stage2_features",
            "model_name": model_name,
            "hf_id": hf_id,
            "init": init,
            "clip_frames": clip_frames,
            "stride": 1,
            "stimuli_summary": summary,
        },
    )
    print(f"[stage2] wrote {out_dir}")
    return out_dir
