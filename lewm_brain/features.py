"""Backbone feature extraction. First model: V-JEPA-2 ViT-L
(`facebook/vjepa2-vitl-fpc64-256`).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _build_clips(pixels: np.ndarray, clip_frames: int, stride: int = 1) -> np.ndarray:
    """Sliding-window clips from (T, H, W) -> (n_clips, clip_frames, H, W)."""
    T = pixels.shape[0]
    if T < clip_frames:
        raise ValueError(f"Movie too short ({T}) for clip size {clip_frames}.")
    starts = np.arange(0, T - clip_frames + 1, stride)
    return np.stack([pixels[s:s + clip_frames] for s in starts], axis=0)


def _to_rgb_uint8(clips: np.ndarray) -> np.ndarray:
    """(N, T, H, W) grayscale uint8 -> (N, T, H, W, 3) uint8."""
    if clips.ndim == 4:
        return np.repeat(clips[..., None], 3, axis=-1).astype(np.uint8, copy=False)
    if clips.ndim == 5 and clips.shape[-1] == 3:
        return clips.astype(np.uint8, copy=False)
    raise ValueError(f"unexpected clip shape {clips.shape}")


def vjepa2_extract_features(
    model: Any,
    processor: Any,
    pixels: np.ndarray,
    clip_frames: int = 64,
    stride: int = 1,
    batch_size: int = 1,
    device: str = "cuda",
    pool: str = "mean",
) -> np.ndarray:
    """Extract per-clip features from V-JEPA-2.

    pixels: (T, H, W) uint8 grayscale OR (T, H, W, 3) uint8 RGB.
    Returns: (n_clips, hidden_size) float32 array.

    Per-frame assignment downstream: clip k spans pixels[k : k+clip_frames];
    we assign its feature to frame `k + clip_frames - 1` (the last frame of
    the clip). The first `clip_frames - 1` frames have no feature and are
    skipped by the encoder.
    """
    import torch

    clips = _build_clips(pixels, clip_frames, stride)
    clips = _to_rgb_uint8(clips)
    n_clips = clips.shape[0]
    print(f"[features] {n_clips} clips of {clip_frames} frames "
          f"({pixels.shape[0]} movie frames, stride {stride})")

    model.eval().to(device)

    feats = []
    with torch.no_grad():
        for i in range(0, n_clips, batch_size):
            batch = list(clips[i:i + batch_size])  # list of (T, H, W, 3) np arrays
            inputs = processor(batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (B, n_patches, D)
            if pool == "mean":
                pooled = hidden.mean(dim=1)
            else:
                raise ValueError(f"unknown pool {pool!r}")
            feats.append(pooled.float().cpu().numpy())
            if (i // batch_size) % 50 == 0:
                print(f"[features]   clip {i}/{n_clips}")

    return np.concatenate(feats, axis=0)


def load_vjepa2(
    hf_id: str,
    init: str = "pretrained",
    seed: int = 0,
    dtype: str = "float32",
):
    """Load (model, processor). `init='random'` returns a freshly-initialized
    model with the same architecture but no pretrained weights, for the
    Brain-Score-style noise-floor control.
    """
    import torch
    from transformers import AutoConfig, AutoModel, AutoVideoProcessor

    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    processor = AutoVideoProcessor.from_pretrained(hf_id)

    if init == "random":
        torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(hf_id)
        model = AutoModel.from_config(config)
        if torch_dtype != torch.float32:
            model = model.to(torch_dtype)
    elif init == "pretrained":
        model = AutoModel.from_pretrained(hf_id, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"unknown init {init!r}")

    return model, processor
