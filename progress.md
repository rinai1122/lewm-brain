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
