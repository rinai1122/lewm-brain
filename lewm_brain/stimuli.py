"""Load natural-movie pixel templates for the Allen Visual Coding stimuli.

The natural-movie templates are the same across all sessions (every mouse
saw the same *Touch of Evil* clips), so they live outside the per-session
NWB. AllenSDK caches them as a separate warehouse fetch.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

_MOVIE_NUMBER = {
    "natural_movie_one": 1,
    "natural_movie_two": 2,
    "natural_movie_three": 3,
}


def get_natural_movie_pixels(cache, stimulus_name: str) -> np.ndarray:
    """Return the (T, H, W) uint8 grayscale movie template.

    Tries AllenSDK's `get_natural_movie_template` first; falls back to a
    direct S3 download of the canonical .npy template if AllenSDK doesn't
    expose that method on this build.
    """
    if stimulus_name not in _MOVIE_NUMBER:
        raise ValueError(f"unknown movie {stimulus_name!r}")
    movie_num = _MOVIE_NUMBER[stimulus_name]

    for method_name in ("get_natural_movie_template", "get_movie_template"):
        if hasattr(cache, method_name):
            try:
                arr = getattr(cache, method_name)(movie_num)
                return np.asarray(arr)
            except Exception as exc:
                print(f"[stimuli] {method_name}({movie_num}) raised {exc!r}; "
                      "falling back to direct S3 download")
                break

    # Fallback: direct download from the public AIBS S3 mirror.
    target_dir = Path("/kaggle/working/movies")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{stimulus_name}.npy"
    if not target.exists() or target.stat().st_size < 1_000_000:
        url = (
            "https://allen-brain-observatory.s3.us-west-2.amazonaws.com/"
            f"visual-coding-neuropixels/ecephys-cache/{stimulus_name}.npy"
        )
        print(f"[stimuli] downloading {url} -> {target}")
        urllib.request.urlretrieve(url, target)
    return np.load(target)
