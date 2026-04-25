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
- V-JEPA / V-JEPA-2
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
No fMRI / CNeuroMod yet. End-to-end pipeline lands on **V-JEPA-2 base + one
session** before any breadth is added.

## Directory layout
TBD — proposed in step 4 of the kickoff plan and pinned here once approved.

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
