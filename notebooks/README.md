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
# Cell 1 — install allensdk from MASTER (PyPI 2.16.2 from Nov 2023
# pins numpy<1.24, which has no Python-3.12 wheel — Kaggle ships 3.12).
# Master drops the numpy pin but its custom NWBFile subclass doesn't
# implement `external_resources`, which is abstract in pynwb >= 2.6 /
# hdmf 4+. Pin pynwb/hdmf to a known-good window in the same resolve.
!pip install --prefer-binary "pynwb<2.6" "hdmf<3.5" \
    git+https://github.com/AllenInstitute/AllenSDK.git
!pip install git+https://github.com/rinai1122/lewm-brain.git
```

**After running Cell 1, restart the kernel** (Run → Restart) so Python
re-imports the now-downgraded `pynwb` / `hdmf`. Then run cells 2 and 3.

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
