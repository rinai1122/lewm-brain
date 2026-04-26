"""Best-effort publish to a Kaggle Dataset (create or version) so partial
outputs survive a kernel crash. Never raises — failed uploads just print a
warning and the local files remain on disk for a manual retry.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def publish_to_kaggle_dataset(
    out_dir: Path,
    dataset_id: str,
    title: str,
    license_name: str = "CC0-1.0",
) -> bool:
    """Push every regular file in `out_dir` to a Kaggle Dataset.

    `dataset_id` must be of the form `<username>/<dataset-slug>` where the
    slug is 6-50 chars, lowercase letters / digits / dashes only.

    Returns True on success, False on failure (does not raise).
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        print(f"[upload] {out_dir} doesn't exist; skipping")
        return False

    upload_dir = out_dir.parent / f".upload_{dataset_id.replace('/', '__')}"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True)
    for f in out_dir.iterdir():
        if f.is_file():
            shutil.copy(f, upload_dir / f.name)

    metadata = {"title": title, "id": dataset_id,
                "licenses": [{"name": license_name}]}
    (upload_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    create = subprocess.run(
        ["kaggle", "datasets", "create",
         "-p", str(upload_dir), "--dir-mode", "zip"],
        capture_output=True, text=True,
    )
    if create.returncode == 0:
        print(f"[upload] created Kaggle Dataset {dataset_id}")
        return True

    blob = (create.stdout + "\n" + create.stderr).lower()
    if "already exists" in blob or "already used" in blob:
        version = subprocess.run(
            ["kaggle", "datasets", "version",
             "-p", str(upload_dir),
             "-m", "stage2 features auto-snapshot",
             "--dir-mode", "zip"],
            capture_output=True, text=True,
        )
        if version.returncode == 0:
            print(f"[upload] versioned Kaggle Dataset {dataset_id}")
            return True
        print(f"[upload] version failed: {version.stderr}")
        return False

    print(f"[upload] create failed: {create.stderr}")
    return False
