# Kaggle notebook recipes

Each pipeline stage is a thin Kaggle notebook that imports the
`lewm_brain` package from this GitHub repo and calls the stage's
orchestrator. All real logic lives in `lewm_brain/` so the notebook
JSON stays trivial.

## One-time Kaggle setup

1. **Account → Phone Verified.** Required for free GPU + internet. ✅
2. **Settings → Account → Show API token.** Not needed for the
   pipeline itself — only if you later want to push notebooks via CLI.

## Stage 1 — Neural data prep

This notebook picks a session, extracts per-frame neural responses for
the natural-movie stimuli, and emits a sanity plot. Saves to a Kaggle
**Notebook Output**, which you'll publish as a Dataset on first run so
later stages can mount it.

Create a new notebook on Kaggle. **Settings:**

- Accelerator: **None** (Stage 1 is CPU-only — Allen download dominates).
- Internet: **On**.
- Persistence: **Files only**.
- Environment: **Always use latest**.

Three cells:

```bash
# Cell 1 — install AllenSDK with --no-deps to bypass its numpy<1.24
# pin (which has no Python-3.12 wheel). Pre-install only the runtime
# deps the cache code actually touches, at versions compatible with
# Kaggle's Py3.12 + numpy 2.x.
!pip install --prefer-binary \
    'pynwb>=2.8,<3' 'hdmf>=3.14,<5' \
    argschema simplejson marshmallow requests-toolbelt tqdm semver \
    ndx-events cachetools nest_asyncio sqlalchemy jinja2
!pip install --no-deps --prefer-binary git+https://github.com/AllenInstitute/AllenSDK.git
!pip install git+https://github.com/rinai1122/lewm-brain.git
```

**Restart the kernel** after Cell 1 (Run → Restart) so the freshly-
installed `lewm_brain` is picked up by the import in Cell 3. Then run
cells 2 and 3.

```python
# Cell 2 — fetch the config from the repo (so we don't pin it twice).
import urllib.request, pathlib
pathlib.Path("/kaggle/working/configs").mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/rinai1122/lewm-brain/main/configs/default.yaml",
    "/kaggle/working/configs/default.yaml",
)
```

```python
# Cell 3 — run stage 1.
from lewm_brain.config import load_config
from lewm_brain.stages import stage1_neural
cfg = load_config("/kaggle/working/configs/default.yaml")
stage1_neural.run(cfg)
```

After it finishes, click **Save Version → Save & Run All (Commit)**.
When the run completes, on the notebook's right sidebar choose
**Output → New Dataset** and name it `lewm-brain-stage1`. Stage 2
(when written) will mount it as input.

Expected first-run wallclock: dominated by AllenSDK downloading the
session's NWB file. Probe size + count are unverified — see
`data_notes.md` "Things to verify on the very first Kaggle run."

If the run fails on the `bin_responses_per_frame` reshape, copy the
error and the printed shapes back here — that means the natural-movie
stimulus table grain isn't one-row-per-frame, and we adjust
`allen_data.bin_responses_per_frame` accordingly.

## Stage 2 — V-JEPA-2 ViT-L feature extraction

Slides a 64-frame window over the `natural_movie_one` (900 frames) and
`natural_movie_three` (3600 frames) pixel templates and extracts a
mean-pooled feature vector per clip with `facebook/vjepa2-vitl-fpc64-256`.
Output is `features__{stim}.npz` per stimulus.

Settings for the new notebook:

- Accelerator: **GPU T4 x2** (or any single T4 — GPU required).
- Internet: **On**.
- Persistence: **Files only**.
- Environment: **Always use latest**.
- No input datasets needed (movie pixels come from AllenSDK).

Three cells:

```bash
# Cell 1 — same install as Stage 1, plus a transformers floor for V-JEPA-2.
!pip install --prefer-binary 'pynwb>=2.8,<3' 'hdmf>=3.14,<5' \
    argschema simplejson marshmallow requests-toolbelt tqdm semver \
    ndx-events cachetools nest_asyncio sqlalchemy jinja2
!pip install --prefer-binary 'transformers>=4.45'
!pip install --no-deps --prefer-binary git+https://github.com/AllenInstitute/AllenSDK.git
!pip install git+https://github.com/rinai1122/lewm-brain.git
```

Restart the kernel after Cell 1 (Run → Restart).

```python
# Cell 2 — fetch config (same as Stage 1).
import urllib.request, pathlib
pathlib.Path("/kaggle/working/configs").mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/rinai1122/lewm-brain/main/configs/default.yaml",
    "/kaggle/working/configs/default.yaml",
)
```

```python
# Cell 3 — run stage 2 with the pretrained V-JEPA-2 ViT-L (model_index=0
# in configs/default.yaml). Re-run with model_index=1 for the random-init
# control after this finishes.
from lewm_brain.config import load_config
from lewm_brain.stages import stage2_features
cfg = load_config("/kaggle/working/configs/default.yaml")
stage2_features.run(cfg, model_index=0)
```

**Expected wallclock**: ~5 min for `natural_movie_one` (837 clips on T4 fp32 batch=1) + ~30 min for `natural_movie_three` (3537 clips). Total ~35 min.

When it finishes: **Save Version → Save & Run All**, then publish output as a Kaggle Dataset named `lewm-brain-stage2`.

Then run the **same notebook again** (or fork it) with Cell 3 changed to:

```python
stage2_features.run(cfg, model_index=1)  # random-init control
```

This produces `vjepa2_vitl_random/features__{stim}.npz` — the
noise-floor control needed for the encoding-model comparison.

### Stage 2 (alt model) — VideoMAE-large

`MCG-NJU/videomae-large` is the pixel-level video MAE baseline (family
(a)). Same notebook scaffold; the model entry at `model_index=2`
(pretrained) and `model_index=3` (random-init) lives in
`configs/default.yaml`. VideoMAE-large's native preset is 16 frames at
224² (positional embeddings are learned for that length — don't try to
force 64), so wallclock is shorter than V-JEPA-2 per stim.

```python
# Cell 3 — VideoMAE pretrained.
from lewm_brain.config import load_config
from lewm_brain.stages import stage2_features
cfg = load_config("/kaggle/working/configs/default.yaml")
stage2_features.run(cfg, model_index=2)
```

Then re-run with `model_index=3` for the random-init control. Outputs
publish to `sungjiwang/lewm-brain-stage2-vmae-l16-tt` and
`…-vmae-l16-tt-rand`.

## Stage 3 — Ridge encoding model

Fits a per-neuron ridge regression from V-JEPA-2 features to neural
responses, scores it on the held-out repeat, and computes split-half
reliability as the noise ceiling. Pure CPU + numpy/scipy linear algebra,
no GPU needed.

Create a new Kaggle notebook. **Settings:**

- Accelerator: **None**.
- Internet: **On** (only needed for the `pip install` from GitHub).
- Persistence: **Files only**.
- Environment: **Always use latest**.
- **Add Inputs**: `sungjiwang/lewm-brain-stage1` and
  `sungjiwang/lewm-brain-stage2`. They mount at
  `/kaggle/input/lewm-brain-stage1` and `/kaggle/input/lewm-brain-stage2`,
  which matches the defaults in `stage3_encoder.run()`.

Three cells:

```bash
# Cell 1 — minimal install. No AllenSDK, no transformers needed for
# stage 3; we only read npz/csv from the mounted datasets.
!pip install git+https://github.com/rinai1122/lewm-brain.git
```

Restart the kernel after Cell 1 (Run → Restart).

```python
# Cell 2 — fetch config (same as earlier stages).
import urllib.request, pathlib
pathlib.Path("/kaggle/working/configs").mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/rinai1122/lewm-brain/main/configs/default.yaml",
    "/kaggle/working/configs/default.yaml",
)
```

```python
# Cell 3 — run stage 3 against the pretrained V-JEPA-2 ViT-L features.
from lewm_brain.config import load_config
from lewm_brain.stages import stage3_encoder
cfg = load_config("/kaggle/working/configs/default.yaml")
stage3_encoder.run(cfg, model_name="vjepa2_vitl")
```

Outputs land at `/kaggle/working/stage3/vjepa2_vitl/` and auto-publish
to `sungjiwang/lewm-brain-stage3`. Re-run with
`model_name="vjepa2_vitl_random"`
once the random-init Stage 2 features are also published, to get the
noise-floor control.

### Stage 3 — VideoMAE encoding

For the VideoMAE pull, **add Inputs** for
`sungjiwang/lewm-brain-stage1` and
`sungjiwang/lewm-brain-stage2-vmae-l16-tt` (or `…-vmae-l16-tt-rand` for
the noise floor). Then:

```python
from pathlib import Path
from lewm_brain.config import load_config
from lewm_brain.stages import stage3_encoder

cfg = load_config("/kaggle/working/configs/default.yaml")
stage3_encoder.run(
    cfg,
    model_name="videomae_large",
    stage1_root=Path("/kaggle/input/lewm-brain-stage1"),
    stage2_root=Path("/kaggle/input/lewm-brain-stage2-vmae-l16-tt"),
)
```

Outputs auto-publish to `sungjiwang/lewm-brain-stage3-vmae` (per-model
slug, configured in `configs/default.yaml` so VideoMAE doesn't clobber
the V-JEPA-2 Stage 3 dataset). Re-run with
`model_name="videomae_large_random"` and `stage2_root` pointing at
`…-vmae-l16-tt-rand` for the noise floor; that one publishes to
`sungjiwang/lewm-brain-stage3-vmae-rand`.

## Stage 4 — Straightening (Hénaff 2021)

Computes per-step trajectory curvature in pixel / model / neural-population
spaces on the same 837-step time grid (`natural_movie_one`), then reports
mean θ per space and the headline Δθ. AllenSDK is needed again because
the pixel template is pulled fresh from the Allen warehouse — the npz
files in Stage 1 / 2 do not store raw pixels.

Create a new Kaggle notebook. **Settings:**

- Accelerator: **None** (CPU-only — no GPU work in Stage 4).
- Internet: **On** (AllenSDK warehouse + the GitHub install).
- Persistence: **Files only**.
- Environment: **Always use latest**.
- **Add Inputs**: `sungjiwang/lewm-brain-stage1` and the Stage 2 dataset
  for the model you want to score (`sungjiwang/lewm-brain-stage2-l16-tt`
  for the pretrained pull, `…-l16-tt-rand` for the random-init control).
  These mount at `/kaggle/input/<slug>/`.

Three cells:

```bash
# Cell 1 — same install as Stage 1 (AllenSDK is needed for pixels).
!pip install --prefer-binary 'pynwb>=2.8,<3' 'hdmf>=3.14,<5' \
    argschema simplejson marshmallow requests-toolbelt tqdm semver \
    ndx-events cachetools nest_asyncio sqlalchemy jinja2
!pip install --no-deps --prefer-binary git+https://github.com/AllenInstitute/AllenSDK.git
!pip install git+https://github.com/rinai1122/lewm-brain.git
```

Restart the kernel after Cell 1 (Run → Restart).

```python
# Cell 2 — fetch config (same as earlier stages).
import urllib.request, pathlib
pathlib.Path("/kaggle/working/configs").mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/rinai1122/lewm-brain/main/configs/default.yaml",
    "/kaggle/working/configs/default.yaml",
)
```

```python
# Cell 3 — run stage 4. Pass the actual mounted input paths if Kaggle
# nests them under /kaggle/input/datasets/sungjiwang/<slug>/ for this
# account (verified in the Stage 3 entry of progress.md).
from pathlib import Path
from lewm_brain.config import load_config
from lewm_brain.stages import stage4_straightening

cfg = load_config("/kaggle/working/configs/default.yaml")
stage4_straightening.run(
    cfg,
    model_name="vjepa2_vitl",
    stage1_root=Path("/kaggle/input/lewm-brain-stage1"),
    stage2_root=Path("/kaggle/input/lewm-brain-stage2-l16-tt"),
)
```

Outputs land at `/kaggle/working/stage4/vjepa2_vitl/` and auto-publish
to `sungjiwang/lewm-brain-stage4`. Re-run with
`model_name="vjepa2_vitl_random"` and `stage2_root` pointing at
`lewm-brain-stage2-l16-tt-rand` for the random-init noise floor.

### Stage 4 — VideoMAE straightening

Same scaffold; swap inputs to the VideoMAE Stage 2 dataset and pass the
matching `model_name`:

```python
from pathlib import Path
from lewm_brain.config import load_config
from lewm_brain.stages import stage4_straightening

cfg = load_config("/kaggle/working/configs/default.yaml")
stage4_straightening.run(
    cfg,
    model_name="videomae_large",
    stage1_root=Path("/kaggle/input/lewm-brain-stage1"),
    stage2_root=Path("/kaggle/input/lewm-brain-stage2-vmae-l16-tt"),
)
```

Auto-publishes to `sungjiwang/lewm-brain-stage4-vmae` (per-model slug;
won't collide with the V-JEPA-2 Stage 4 dataset). Re-run with
`model_name="videomae_large_random"` and `stage2_root` pointing at
`…-vmae-l16-tt-rand` to publish to `…-stage4-vmae-rand`.

Headline numbers to eyeball in the printed log:

- `Δ_pix-model` — pretrained model should make the trajectory
  measurably straighter than raw pixels (Hénaff prediction; positive Δ).
  Random-init should make it *less* straight — its representation is
  effectively a random rotation of pixels, so curvature should match
  pixel curvature within noise.
- `Δ_pix-neural` — Hénaff 2021 §Results headline: cortex straightens
  natural-video trajectories. We're reproducing this in mouse VIS,
  which is a weaker prediction than their primate V1 result but worth
  measuring.
