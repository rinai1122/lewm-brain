"""Best-effort publish to a Kaggle Dataset (create or version) so partial
outputs survive a kernel crash. Never raises — failed uploads just print a
warning and the local files remain on disk for a manual retry.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


# kaggle CLI 2.0.0 has a habit of returning exit code 0 even when it prints
# "Dataset creation error: Invalid Owner Id" (or similar) to stdout. Treat
# those as failures so we don't claim success on a no-op upload.
_ERROR_MARKERS = (
    "dataset creation error",
    "dataset version error",
    "invalid owner id",
    "403 - forbidden",
    "404 - not found",
)


def _looks_like_error(text: str) -> bool:
    lower = text.lower()
    return any(m in lower for m in _ERROR_MARKERS)


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
    create_blob = create.stdout + "\n" + create.stderr
    if create.returncode == 0 and not _looks_like_error(create_blob):
        print(f"[upload] created Kaggle Dataset {dataset_id}")
        return True

    # Kaggle CLI 2.0.x reports the "this dataset already exists" condition
    # via three different strings depending on whether the slug, the title,
    # or the URL collides. "is already in use" is the title-collision form.
    lower = create_blob.lower()
    already_exists = (
        "already exists" in lower
        or "already used" in lower
        or "is already in use" in lower
    )
    if already_exists:
        version = subprocess.run(
            ["kaggle", "datasets", "version",
             "-p", str(upload_dir),
             "-m", f"{dataset_id} auto-snapshot",
             "--dir-mode", "zip"],
            capture_output=True, text=True,
        )
        version_blob = version.stdout + "\n" + version.stderr
        if version.returncode == 0 and not _looks_like_error(version_blob):
            print(f"[upload] versioned Kaggle Dataset {dataset_id}")
            return True
        print(
            f"[upload] version FAILED for {dataset_id} "
            f"(returncode={version.returncode})\n"
            f"--- stdout ---\n{version.stdout}\n"
            f"--- stderr ---\n{version.stderr}"
        )
        return False

    print(
        f"[upload] create FAILED for {dataset_id} "
        f"(returncode={create.returncode})\n"
        f"--- stdout ---\n{create.stdout}\n"
        f"--- stderr ---\n{create.stderr}"
    )
    return False
