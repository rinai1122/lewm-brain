# LeWM-Brain — testing latent-predictive learning against neural data

## Project goal
Test whether JEPA-style latent-predictive world models (LeWM, V-JEPA, I-JEPA)
explain neural responses to naturalistic video better than four comparison
baseline families:

- (a) pixel-level predictive models
- (b) contrastive SSL
- (c) supervised classifiers
- (d) random-init baselines

Two convergent analyses on the same neural data:

1. **Neural predictivity** — linear encoding model fit per neuron via ridge
   regression on model features. Reference: Schrimpf et al. 2018 (Brain-Score).
2. **Perceptual / neural straightening** — temporal-trajectory curvature in
   model representation space and in neural population activity, following
   Hénaff et al. 2021 *Nat. Neurosci.*

## Models

**Focal class — latent-predictive:**
- LeWM
- V-JEPA / V-JEPA-2 (note: smallest released V-JEPA-2 backbone is **ViT-L**
  at 0.3 B params — `facebook/vjepa2-vitl-fpc64-256`, Apache 2.0, ungated.
  No ViT-B "base" checkpoint exists; references to "base" in this repo mean
  this ViT-L variant.)
- I-JEPA

**Comparison families:**
- (a) Pixel-level predictive — e.g. MAE / VideoMAE
- (b) Contrastive SSL — e.g. SimCLR, MoCo (specific checkpoints TBD)
- (c) Supervised — e.g. ImageNet-trained ViT, Kinetics-trained video classifier
- (d) Random-init baselines — same architectures, untrained weights

Specific checkpoint identities + hashes pinned in a model registry once chosen.
No model is trained from scratch in this project.

## Analyses
Both analyses run on the *same* held-out stimulus repeats so the two scores
are directly comparable per neuron / per population.

## Data scope (this session)
Mouse Allen Brain Observatory **Neuropixels** natural-movie sessions only.
No fMRI / CNeuroMod yet. End-to-end pipeline lands on **V-JEPA-2 ViT-L
(`facebook/vjepa2-vitl-fpc64-256`) + one session** before any breadth is
added.

## Compute & storage strategy
Local disk and GPU are too constrained to be the runtime — the laptop has
9 GB free on C: and a 2 GB-VRAM MX450. So:

- **Local laptop = editor + git only.** No `torch`, no `allensdk`, no
  project venv on the laptop. All Python installs and runs happen on
  Kaggle.
- **Kaggle Notebooks = compute.** Free T4 (16 GB VRAM) / P100, ~30 hr/week
  GPU quota, internet enabled (gated by phone verification on the Kaggle
  account). One notebook per pipeline stage.
- **Data = streamed from DANDI**, not cached locally. Allen Neuropixels
  sessions are mirrored on DANDI as NWB; we open them via the
  `pynwb` + `fsspec`/`remfile` streaming path so no full session ever
  lands on disk. Exact incantation to be verified in step 3.
- **Intermediates = a single Kaggle Dataset owned by the user.** Each
  pipeline stage writes its outputs (extracted features, ridge weights,
  per-neuron scores) as a new version of the dataset; the next stage
  reads them as input. This makes the pipeline naturally resumable
  across the ~12 hr Kaggle session limit and free of local-disk
  pressure.
- **Repro discipline.** Every notebook pins package versions, sets
  every seed, and writes its resolved config + commit hash next to its
  outputs in the dataset.

Implication for development loop: every iteration is `git push` →
re-run the affected Kaggle notebook. There is no fast local sanity-check
for anything that needs torch or Allen data; debugging happens on
Kaggle.

## Directory layout
See `pipeline.md` for the full proposal. Headline:

```
lewm-brain/
├── CLAUDE.md / data_notes.md / pipeline.md / progress.md
├── pyproject.toml / requirements-kaggle.txt
├── lewm_brain/        # importable package — all real logic
│   ├── config.py / allen_data.py / stimuli.py / features.py
│   ├── encoding.py / straightening.py
│   └── stages/        # one orchestrator per pipeline stage
├── notebooks/         # 3-cell Kaggle shims (`pip install` + `stage.run(cfg)`)
└── configs/           # YAML config consumed by every stage
```

## Research-design defaults (from `pipeline.md`, overridable in `configs/default.yaml`)
1. **Held-out repeats** — last repeat as global test; LOO-repeat CV for ridge α.
2. **Running speed / pupil** — included as additional regressors; not regressed-out.
3. **Cortical-depth pooling** — pool within area (no layer stratification).
4. **Sanity-check session** — deterministic: BO 1.1 + VISp + ≥1 higher area, max
   good-units; tie-break low session_id.

## Constraints (load-bearing — read before scaling any analysis)

- **Reproducibility.** Pin every dependency version. Set every seed. Write
  the resolved config next to every output artifact.
- **Honesty.** If something fails or is uncertain, say so explicitly. Never
  fabricate numbers, paths, or API behavior. Read the docs / source rather
  than guess.
- **Sanity-check before scaling.** Any analysis that will eventually run on
  all sessions / all models must first run on one session / one model and
  produce a plot that can be eyeballed.
- **Save intermediates.** Extracted features, fitted regression weights, and
  per-neuron scores all go to disk. Re-running should be cheap.

## Working agreement

- One **feature branch per analysis**. Commit at every working checkpoint
  with descriptive messages.
- After each substantive step, append a one-paragraph entry to `progress.md`:
  what worked, what didn't, decisions made that the user should know.
- **Stop and ask** on *research-design* ambiguity. Examples: which stimulus
  repeats are held-out, how to handle running speed as a covariate, which
  cortical layers / depths to pool. **Do not stop** for code-shape choices
  (file paths, library calls, plot styling) — make a reasonable choice and
  note it.
- When implementing a non-obvious analytical choice, cite paper + section
  in a code comment (e.g. `# ridge alpha by LOO-CV per neuron, Schrimpf
  et al. 2018 §2.3`).

## Out of scope (this project, for now)
- Training models from scratch; hyperparameter sweeps.
- Human fMRI experiments.
- Violation-of-expectation experiment.
- Paper writing.
