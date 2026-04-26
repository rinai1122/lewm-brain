"""Config loading + path resolution.

Stage notebooks are 3-cell shims that do:
    cfg = load_config('/kaggle/input/lewm-brain-source/configs/default.yaml')
    stage1.run(cfg)

so this module owns the schema and the I/O conventions.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Kaggle conventions. Override via env vars when running off-Kaggle (tests).
KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
KAGGLE_WORKING = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))


@dataclass
class Config:
    """Mutable config bag — loaded from YAML, mutated as defaults resolve."""
    raw: dict[str, Any]

    def get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set(self, *path: str, value: Any) -> None:
        node = self.raw
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    def dump(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        return Config(raw=yaml.safe_load(f))


def git_commit_hash() -> str | None:
    """Best-effort: returns short commit hash if running inside a clone."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def write_artifact_manifest(out_dir: Path, cfg: Config, extra: dict[str, Any]) -> None:
    """Drop a config.json next to every output. Captures commit hash and
    any caller-supplied provenance (input dataset versions, library
    versions, runtime).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": cfg.raw,
        "git_commit": git_commit_hash(),
        **extra,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
