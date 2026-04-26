# Pipeline plan (step 4 deliverable)

## Repo layout

```
lewm-brain/
├── CLAUDE.md             # project charter — load-bearing, read first
├── data_notes.md         # AllenSDK / DANDI / pynwb working notes
├── pipeline.md           # this file — directory + pipeline plan
├── progress.md           # rolling progress log
├── pyproject.toml        # pip-installable package (`pip install git+…`)
├── requirements-kaggle.txt  # extra pins beyond the Kaggle base image
├── lewm_brain/           # shared Python package
│   ├── __init__.py
│   ├── config.py         # paths, seeds, version constants
│   ├── allen_data.py     # session selection + neural extraction
│   ├── stimuli.py        # frame-aligned movie clip handling   (stage 2)
│   ├── features.py       # backbone feature extraction         (stage 2)
│   ├── encoding.py       # ridge encoding model                (stage 3)
│   ├── straightening.py  # Hénaff curvature                    (stage 4)
│   └── stages/           # orchestrators — each one a Kaggle notebook target
│       ├── stage1_neural.py
│       ├── stage2_features.py
│       ├── stage3_encoder.py
│       └── stage4_straightening.py
├── notebooks/
│   └── README.md         # the 3-cell shim each Kaggle notebook contains
└── configs/
    └── default.yaml      # session_id, model_id, seeds, alpha grid, …
```

The notebooks themselves are **3-cell shims** (`!pip install …`,
`from lewm_brain.stages import stageN`, `stageN.run(cfg)`). All real
logic lives in the `lewm_brain/` package so we can lint/test/diff it
like normal Python — `.ipynb` JSON never gets hand-edited.

## Defaults for the four research-design questions

These are my picks so we can move; flip any of them by editing
`configs/default.yaml` and re-running the affected stage.

1. **Held-out repeats.** **Last repeat of each movie is the global test
   set.** Ridge α is selected on the remaining repeats by leave-one-
   repeat-out CV per neuron (Schrimpf et al. 2018 §2.3 style). Reason:
   simple, replicable, and matches what most Brain-Score-adjacent papers
   do for movie stimuli.
2. **Running speed and pupil.** **Include as additional regressors** in
   the encoding-model design matrix (not regressed out beforehand, not
   filtered to stationary). Reason: regressing-out throws away variance
   that may be shared with stimulus drive; filtering loses too much data
   (mice run a lot). Reported neural predictivity is the contribution
   of the *visual* regressors over and above behavior.
3. **Cortical-depth pooling.** **Pool all good units within an area**
   for v1 of the analysis. Layer stratification is deferred — it's a
   v2 question and the per-layer unit counts will be too low for
   stable per-neuron ridge in many sessions.
4. **Single-session sanity-check pick.** **Deterministic rule**: among
   `session_type == 'brain_observatory_1.1'` sessions with VISp **and**
   ≥ 1 of {VISl, VISal, VISpm, VISam, VISrl}, pick the one with the
   highest count of units passing the default unit filters in those
   areas. Tie-break by lowest `session_id`. The rule + chosen
   `session_id` are written to `configs/default.yaml` on first run.

If any of these is wrong, say so; I won't rewrite results around the
default once we have them, but flipping early is cheap.

## Pipeline stages

Each stage is one Kaggle notebook, owns one Kaggle Dataset version, and
is independently re-runnable. Outputs include the resolved config and
the git commit hash so any artifact can be traced back to its source.

### Stage 1 — Neural data prep
Input:  AllenSDK warehouse (or DANDI:000021 fallback).
Output: `stage1/{session_id}/neural.npz` with arrays
        `responses (n_repeats × n_frames × n_units)`,
        `unit_meta` (id, area, depth, quality metrics),
        `presentation_meta` (start_time, repeat, frame),
        `running_speed (n_repeats × n_frames)`,
        plus `config.json` and `sanity.png`.
Notebook: `notebooks/01_neural_prep` shim.
Sanity plot: per-area unit count + a (units × frames) heatmap for one
repeat of `natural_movie_one`.

### Stage 2 — Backbone feature extraction
Input:  Stage 1 dataset (for the per-frame timing) + the natural-movie
        video files (re-rendered from AllenSDK at 30 fps).
Output: `stage2/{model_id}/features.npz` with
        `features (n_frames × d_model)` per layer of interest,
        plus `config.json` (model id, layer, clip-window choice).
Notebook: `notebooks/02_features_vjepa2` shim — first model is
`facebook/vjepa2-vitl-fpc64-256`. Other models swap the model id.
**Open design choice surfaced here:** the V-JEPA-2 ViT-L expects
64-frame clips; movie frames are 900 / 3600 long. Options for the
clip-window are sliding-stride-1, non-overlapping, or center-crop. We'll
default to **sliding stride 1 with output assigned to the last frame of
the clip** unless you prefer otherwise.

### Stage 3 — Ridge encoding model
Input:  Stage 1 + Stage 2 datasets.
Output: `stage3/{model_id}/encoding.npz` with `pearson_r (n_units,)`
        on the held-out repeat, fitted weights, and chosen α per unit.
Notebook: `notebooks/03_encoding` shim. Always co-runs against a
random-init backbone of the same architecture for the noise-floor
control (Schrimpf et al. 2018 §3).

### Stage 4 — Straightening
Input:  Stage 1 + Stage 2 datasets.
Output: `stage4/{model_id}/straightening.npz` with per-clip curvature
        in pixel space, in model representation space, and in the
        neural population space.
Notebook: `notebooks/04_straightening` shim. Method follows Hénaff
et al. 2021 *Nat. Neurosci.* §Methods.

### Stage 5 — Cross-model summary (later)
Aggregates stages 3+4 across all model families and produces the
headline figures.

## Sequencing for "one model end-to-end"

Stage 1 → Stage 2 (V-JEPA-2 ViT-L) → Stage 3 → Stage 4 on **one
session, one model**. After this works and the sanity plot looks
reasonable, expand to (i) all sessions for V-JEPA-2 ViT-L, then
(ii) the other model families one at a time.
