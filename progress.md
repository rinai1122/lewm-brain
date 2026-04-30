# Progress log

## 2026-04-25 — Step 1: project charter
Authored `CLAUDE.md` with the project goal, focal-vs-comparison model split
(JEPA family as focal; pixel-predictive / contrastive SSL / supervised /
random-init as the four baseline families), both analyses (Brain-Score ridge
encoding, Hénaff straightening), session scope (Allen Neuropixels first,
V-JEPA-2 base end-to-end before any breadth), and the load-bearing
constraints. Initialized a fresh git repo at the project root (it had been
sitting inside a stray home-directory git repo). Directory layout left as
TBD pending step 4.

## 2026-04-25 — Step 2: environment survey (read-only)
Surveyed the environment without installing anything.

**Python.** System Python 3.12.10 at
`C:\Users\sungj\AppData\Local\Programs\Python\Python312\python.exe`. No
conda detected (`conda` not on PATH). No project venv yet.

**Installed packages (system Python).** Effectively bare: `numpy 2.4.4`,
`pillow 12.2.0`, `mpmath`, `sympy`, `pip 25.0.1`. None of `torch`,
`torchvision`, `transformers`, `timm`, `scipy`, `scikit-learn`, `xarray`,
`h5py`, `allensdk`, `brainio`, `brain-score`, `pynwb`, `dandi` are
installed.

**GPU.** NVIDIA GeForce MX450, 2 GB VRAM, driver 551.61, CUDA 12.4. This is
an entry-level laptop GPU. V-JEPA-2 ViT-L weights alone (≈ 1 GB+ fp16) plus
activations will not comfortably fit; V-JEPA-2 ViT-B (~85 M params) might
fit at small batch / sequence length, otherwise CPU inference is the
fallback.

**Disk.** C: drive is **99 % full — only 9.0 GB free of 458 GB**. This is a
hard blocker for both pip installs (torch wheels alone are ~2–3 GB) and
Allen NWB session downloads (single sessions are typically 5–50 GB).
Resolution required before step 3's downstream work can run.

**V-JEPA repo.** Not cloned anywhere obvious under `~/Desktop/Sample/` or
`~/`. Will be cloned when needed.

**Decisions / flags for the user:**
- Python 3.12 may collide with `allensdk` (historically conservative on
  Python versions). To be verified against AllenSDK release notes during
  step 3 before we settle on an interpreter version for the project venv.
- The disk-space situation needs a decision before any installs or data
  pulls. Options: free space on C:, target a different drive for both
  the venv and the data cache, or use an external/cloud volume.
- No installs were performed. All findings are inventory only.

## 2026-04-25 — disk/GPU constraint resolved → Kaggle-only strategy
User can't free local disk or upgrade GPU. Locked in a Kaggle-only
runtime: laptop = editor + git only; all installs and runs on Kaggle
Notebooks (T4/P100, 30 hr/week GPU); intermediates in a single
user-owned Kaggle Dataset; data streamed/cached one session at a time.
GitHub repo created (private, `rinai1122/lewm-brain`) and the existing
commits pushed. Verified V-JEPA-2 is **not** gated (Apache 2.0); HF
account dropped from the prep checklist. Smallest released V-JEPA-2 is
ViT-L 0.3 B (`facebook/vjepa2-vitl-fpc64-256`); there is no ViT-B
"base" — references to "base" in this repo now mean ViT-L. Kaggle
account setup paused on the user's side pending phone access for
verification.

## 2026-04-25 — Step 3: data_notes.md
Wrote `data_notes.md` from AllenSDK + DANDI + pynwb docs. Verified:
session indexing via `EcephysProjectCache.get_session_table` on
`session_id` × `ecephys_structure_acronyms`; six target visual cortical
areas (VISp + VISl/VISal/VISpm/VISam/VISrl); two stimulus sets are
mirrored on DANDI as `000021` (Brain Observatory 1.1, 32 mice / 477 GB)
and `000022` (Functional Connectivity, 375 GB) — and only BO 1.1 has
`natural_movie_three`; AllenSDK's three default unit filters
(`isi_violations < 0.5`, `amplitude_cutoff < 0.1`,
`presence_ratio > 0.9`); pynwb ROS3 streaming is preferred over fsspec
but **needs conda's `h5py`**, not the PyPI build. Python compat:
PyPI's latest `allensdk` is 2.16.2 (2023-11) with classifiers up to
3.11; master branch declares 3.10–3.13. Plan is to target whatever
Kaggle ships and pin from there. Items I deliberately did **not**
guess at — exact movie repeat counts, exact frame counts per repeat,
Kaggle Python version, ROS3-on-Kaggle viability — are listed in
`data_notes.md` under "Things to verify on the very first Kaggle run."
Four research-design questions surfaced for user input (held-out
repeats, running-speed handling, depth pooling, single-session pick).

## 2026-04-26 — Stage 1 runs end-to-end on Kaggle ✓
First successful end-to-end Kaggle run on session **798911424** (BO 1.1):
- 477 good visual-cortex units passing the default filter.
- Per-area: VISp 100, VISl 78, VISal 98, VISpm 0, VISam 149, VISrl 52.
- `natural_movie_one`: responses (20, 900, 477), running (20, 900).
- `natural_movie_three`: responses (10, 3600, 477), running (10, 3600).
- NWB ready in 146.5 s (download + open).

What it took to get here (so future-me doesn't repeat the fight):
1. PyPI `allensdk==2.16.2` (Nov 2023) pins `numpy<1.24` → fails to build on
   Kaggle's Py3.12.
2. AllenSDK `master` branch pins `pynwb<2.6` and `hdmf<3.5`, which transitively
   pull `numpy<1.24` → same failure under standard `pip install`.
3. The actual install that works: pre-install modern `pynwb>=2.8,<3` /
   `hdmf>=3.14,<5` plus AllenSDK's other runtime deps explicitly, then
   `pip install --no-deps git+…/AllenSDK.git`. The `--no-deps` skips
   AllenSDK's old version pins entirely.
4. AllenSDK's pynwb-based session loader still trips on the
   `NWBFile.external_resources` abstract-method mismatch — bypassed by
   monkey-patching `EcephysSession.__init__` to force `test=False` and
   reading the NWB file directly with `h5py`.

Verified shapes folded into `data_notes.md`. Notebook recipe in
`notebooks/README.md` updated to the working install. Next: publish the
output as a Kaggle Dataset (`lewm-brain-stage1`) so Stage 2 can mount it,
then write Stage 2 (V-JEPA-2 ViT-L feature extraction) against the
verified per-frame timing.

## 2026-04-26 — Step 4 + Stage 1 implementation
User approved Kaggle account verification and said "do everything,"
which I read as: skip the step-4 hold-and-wait gate, pick reasonable
defaults for the four open research-design questions, and ship the
sanity-check pipeline. Wrote `pipeline.md` with the full directory
proposal and the four defaults (last-repeat held-out, running-speed
as covariate, pool within area, deterministic session pick by max
good-units in VISp + ≥1 higher area). Built the `lewm_brain` package
skeleton (`config.py`, `allen_data.py`, `stages/stage1_neural.py`)
plus `pyproject.toml`, `requirements-kaggle.txt`, and
`configs/default.yaml`. Stage 1 picks a session, extracts good-unit
visual-cortex spikes per movie frame for `natural_movie_one` and
`natural_movie_three`, computes per-frame running speed, writes an
.npz + a sanity heatmap + a config.json with the resolved
session_id and commit hash. Stages 2–4 deliberately not yet
implemented — Stage 2's reshape will depend on what shape Stage 1
actually produces, and writing it ahead of the first Kaggle run
would be guessing. Notebook recipe documented in `notebooks/README.md`.
Local syntax check: all .py files compile under Python 3.12.

## 2026-04-30 — Kaggle Dataset uploads were silently failing for weeks
User couldn't find `lewm-brain-stage2` on Kaggle despite the
`[upload] created Kaggle Dataset rinai1122/lewm-brain-stage2`
success message at the end of every Stage 2 run. Diagnosis: the kaggle CLI
2.0.0 was returning **exit code 0 while printing
`Dataset creation error: Invalid Owner Id` to stdout**, and
`kaggle_io.publish_to_kaggle_dataset` only inspected returncode. Two
upstream causes: (1) a previous session plugged in `rinai1122` (the user's
GitHub handle) as the Kaggle owner — the actual Kaggle username is
`sungjiwang`; (2) Stage 1 had no auto-publish hook at all, so even with
the right owner it would never have appeared. Fixes:
- `configs/default.yaml`: dataset_id slugs flipped to `sungjiwang/...`;
  added a `stage1.dataset_id` block.
- `lewm_brain/kaggle_io.py`: scan stdout/stderr for `Dataset creation
  error`, `Invalid Owner Id`, etc., even on returncode 0; print full
  stdout/stderr on every failure path so future regressions are visible.
- `lewm_brain/stages/stage1_neural.py`: mirror Stage 2's per-stim
  `kaggle_io.publish_to_kaggle_dataset` call so Stage 1 outputs are durable
  on Kaggle.
Effect: no Stage 1 / Stage 2 dataset has ever existed under `sungjiwang`
on Kaggle, so the user needs to re-run Stage 1 (CPU, ~3 min) and
re-publish Stage 2 (either manually from the existing Stage 2 notebook's
saved output with the corrected owner, or by re-running Stage 2) before
Stage 3 can find its inputs.
