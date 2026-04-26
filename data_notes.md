# Data notes — Allen Brain Observatory Neuropixels (Visual Coding)

Working notes for accessing the Allen Visual Coding – Neuropixels dataset.
Verified items have a citation; unverified items are tagged **`[VERIFY]`**
and will be confirmed on the first Kaggle run rather than guessed at here.

## What the dataset is

Mouse extracellular electrophysiology with Neuropixels probes, recorded by
the Allen Institute, intended to mirror the two-photon Brain Observatory
visual-stimulus protocol. ~100 k cortical units across all sessions. Six
visual cortical areas are targeted with the probes: **VISp (V1), VISl,
VISal, VISpm, VISam, VISrl**, plus subcortical structures (LGN, LP,
hippocampal formation). [brain-map.org/circuits-behavior/visual-coding-neuropixels]

## Two session types

The release is split into two stimulus sets, run in separate sessions on
separate cohorts of mice:

- **`brain_observatory_1.1`** — designed to align with the calcium-imaging
  Brain Observatory stimulus set. Includes `natural_movie_one` and
  `natural_movie_three`, plus drifting/static gratings, natural scenes,
  flashes, gabors, and spontaneous epochs. **`[VERIFY]`** repeat counts.
- **`functional_connectivity`** — designed for higher-fidelity stimulus
  repetition. Includes `natural_movie_one` (more repeats than in BO 1.1),
  but **does not include `natural_movie_three`**. **`[VERIFY]`** the
  exact repeat count.

For our two analyses (encoding-model and Hénaff-style straightening),
**both session types are usable for `natural_movie_one`**; only
`brain_observatory_1.1` is usable for `natural_movie_three`.

[Whitepaper: brain-map.org/circuits-behavior/visual-coding-neuropixels;
DANDI: 000021 (BO 1.1) and 000022 (FC) — see "Hosting" below.]

## Session indexing

The canonical entry point is `EcephysProjectCache`:

```python
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
cache = EcephysProjectCache.from_warehouse(manifest=MANIFEST_PATH)
sessions = cache.get_session_table()
```

`get_session_table()` returns a pandas DataFrame indexed by `session_id`
with columns including `session_type`, `full_genotype`, `age_in_days`,
`sex`, and **`ecephys_structure_acronyms`** — a list of CCFv3 area
acronyms recorded in that session. This is the column to filter on for
visual-cortex coverage.
[allensdk.readthedocs.io ecephys_data_access notebook]

Filtering for sessions with at least V1 and one higher visual area:

```python
viscort = ['VISp','VISl','VISal','VISpm','VISam','VISrl']
def has_v1_plus(acronyms):
    return ('VISp' in acronyms) and any(a in acronyms for a in viscort if a != 'VISp')
chosen = sessions[sessions.ecephys_structure_acronyms.map(has_v1_plus)]
```

Loading one session:

```python
session = cache.get_session_data(session_id)
```

Total mice, BO 1.1: **32**. [DANDI:000021 metadata]
Total assets, BO 1.1: 214 files / 477 GB.
Total assets, FC: 169 files / 375 GB. [DANDI:000022 metadata]

## Stimulus structure (natural movies)

The natural-movie stimuli are clips edited from *Touch of Evil*
(Welles 1958), as in the calcium Brain Observatory.

**Verified on session 798911424 (BO 1.1) — 2026-04-26:**
- Frame rate: **30 fps**.
- `natural_movie_one`: **900 frames/repeat × 20 repeats** = 18 000 rows in
  `intervals/natural_movie_one_presentations`.
- `natural_movie_three`: **3600 frames/repeat × 10 repeats** = 36 000 rows
  in `intervals/natural_movie_three_presentations`.
- Repeats are not stored as a column; we derive them by counting `frame == 0`
  occurrences in the presentation table sorted by `start_time`.

Pulling the presentation table for one stimulus:

```python
pres = session.get_stimulus_table('natural_movie_one')
# columns include: start_time, stop_time, frame, repeat, stimulus_block, ...
```
[allensdk.readthedocs.io ecephys_quickstart]

## Aligning spikes to frames

Two equivalent APIs:

```python
# Spike counts in fixed bins, aligned to each stimulus presentation
counts = session.presentationwise_spike_counts(
    bin_edges=np.arange(0, 0.0334 + 1/30, 1/30),  # one bin per movie frame
    stimulus_presentation_ids=pres.index,
    unit_ids=good_unit_ids,
)  # returns xarray DataArray: (presentation_id, time_relative, unit_id)

# Or: raw spike times annotated by presentation+unit
spikes = session.presentationwise_spike_times(...)
```
[allensdk.readthedocs.io ecephys_quickstart, ecephys_session]

Concrete plan for the encoding model: bin at the movie frame rate (1/30 s)
so each presentation × frame becomes one (n_units,) response vector. For
straightening, the same per-frame binning applies.

## Standard preprocessing — units

AllenSDK's three default unit-quality filters (the ones that are
applied unless explicitly disabled):

| metric              | threshold |
|---------------------|-----------|
| `isi_violations`    | < 0.5     |
| `amplitude_cutoff`  | < 0.1     |
| `presence_ratio`    | > 0.9     |

[allensdk.readthedocs.io ecephys_quality_metrics]

Other metrics (`snr`, `firing_rate`, `nn_hit_rate`, `isolation_distance`)
are documented but the Allen team explicitly does not recommend any of
them as standalone filters. Apply the three defaults; record the unit
count before/after for transparency.

The docs do **not** prescribe a behavioral-covariate (running speed,
pupil) regression. That's a research-design call — see the open
questions below.

## Hosting and streaming

Two ways to get the bytes onto Kaggle:

1. **AllenSDK warehouse / S3 cache** (`EcephysProjectCache.from_s3_cache`
   or `from_warehouse`). Downloads NWB files to a local cache dir. On
   Kaggle this means the ~20 GB notebook scratch, one session at a time.
   Released sizes per session vary; BO 1.1 averages ~2.2 GB/file across
   214 files / 32 mice (so a session with multiple probes is several GB).

2. **DANDI mirror via remote NWB**. Both sessions sets are on DANDI:
   - **`DANDI:000021`** — Brain Observatory 1.1 stimulus set
   - **`DANDI:000022`** — Functional Connectivity stimulus set
   Files are NWB on S3. Streaming via `pynwb` is supported either with
   the **ROS3** HDF5 driver (preferred per pynwb docs — slicing only
   downloads the sliced bytes) or with `fsspec` (works but slower for
   HDF5). ROS3 needs the **conda** `h5py` build, since "Pre-built h5py
   packages on PyPI do not include this S3 support." [pynwb streaming
   tutorial]

```python
from pynwb import NWBHDF5IO
with NWBHDF5IO(s3_url, mode='r', driver='ros3') as io:
    nwbfile = io.read()
```

**Decision for the pipeline:** start with the AllenSDK cache path on
Kaggle (well-trodden, less surprise). Treat ROS3 streaming as a fallback
if a session won't fit in scratch. Record which path was used in the
config that lands next to each output.

## Python / package compat

- Latest **PyPI release of `allensdk` is 2.16.2 (2023-11-30)**, with
  classifiers `Python 3.8 / 3.9 / 3.10 / 3.11` and no `requires_python`
  declared. Python **3.12 is not in that release's tested matrix**.
- The **`master` branch** of `AllenInstitute/AllenSDK` has moved on:
  `pyproject.toml` declares `requires-python = ">=3.10"` and adds
  classifiers for **3.12 and 3.13**. So 3.12 support exists in source
  but is not on PyPI yet at time of writing.
- **Plan**: target whatever Python the Kaggle T4/P100 image ships with,
  but assume 3.10 or 3.11 is the safe sweet spot. **`[VERIFY]`** the
  Kaggle notebook Python version on first run, and pin
  `allensdk==2.16.2` if it works on that interpreter; otherwise install
  master via `pip install git+https://github.com/AllenInstitute/AllenSDK.git`.

**Verified 2026-04-26 on first Kaggle run:** the current Kaggle base
image is **Python 3.12**, with `numpy 2.0.2`, `pandas 2.3.3`,
`h5py 3.15.1` preinstalled. PyPI `allensdk==2.16.2` **does not
install** on this image — its `numpy<1.24` pin forces pip to a 1.23.5
sdist, which has no 3.12 wheel and fails to build from source. **Use
the master branch instead**: `pip install git+https://github.com/AllenInstitute/AllenSDK.git`.
This is what `notebooks/README.md` Cell 1 does.

## Things to verify on the very first Kaggle run

These are the items I refused to guess at — they're easy to check in
~5 minutes once we have a notebook:

1. Kaggle's default Python version + GPU model + scratch-disk size.
2. `allensdk==2.16.2` installs and imports cleanly on that interpreter.
3. `cache.get_session_table()` actually returns and the column names
   match what's documented above.
4. For one BO 1.1 session: number of `natural_movie_one` and
   `natural_movie_three` presentations, their `repeat` values, and the
   per-frame `start_time` deltas (to confirm 30 fps, 900 / 3600 frames).
5. The default unit filter actually drops the units I expect.
6. Whether ROS3 streaming works in the Kaggle image (it requires the
   conda `h5py` build; if Kaggle ships pip `h5py`, ROS3 will fail and
   we fall back to `from_s3_cache`).

## Research-design questions to flag (not for me to decide)

1. **Held-out repeats.** Each session shows multiple repeats of each
   movie. Standard practice in the encoding-model literature is to use
   one or more repeats as a held-out test set, with cross-validation on
   the rest. Which repeat-split? (E.g. last-repeat-as-test; leave-one-
   repeat-out CV; nested CV.)
2. **Running-speed and pupil covariates.** Allen sessions are recorded
   on a head-fixed running wheel; running modulates V1 strongly. Should
   we (a) regress running speed out of neural responses before fitting,
   (b) include it as an additional regressor in the encoding model, or
   (c) restrict to stationary epochs? Each gives a different bias.
3. **Cortical depth pooling.** Each probe samples multiple cortical
   layers in a column. Should we pool across all units in an area, or
   stratify by layer (using the CCFv3 layer assignment AllenSDK
   provides)?
4. **Single-session sanity-check choice.** Among sessions that have V1
   + at least one higher visual area + clean unit yield, do you want
   a deterministic choice (e.g. "the BO 1.1 session with the highest
   total good-unit count") or an arbitrary one? The choice affects
   whether the sanity-check is replicable across collaborators.
