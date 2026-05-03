# Results — comparison families

Drop numbers in as Kaggle runs come back. Existing V-JEPA-2 ViT-L numbers
copied from `progress.md` (2026-05-02) for reference; VideoMAE-large rows
are placeholders pending the 4 Kaggle runs queued on 2026-05-03.

Session: **798911424** (BO 1.1) · stim: **`natural_movie_one`** ·
features: **layer 16, last-tubelet pool**.

Test-clip split-half reliability (noise ceiling): **0.230**.


## Stage 3 — ridge encoding, per-unit Pearson r on held-out clips

| Model                       | Init       | r mean | r median | VISp   | VISal  | VISam | VISl   | VISrl |
|-----------------------------|------------|--------|----------|--------|--------|-------|--------|-------|
| V-JEPA-2 ViT-L              | pretrained | 0.071  | 0.047    | 0.072  | 0.019  | 0.108 | 0.073  | 0.064 |
| V-JEPA-2 ViT-L              | random     | 0.021  | 0.015    | -0.007 | 0.015  | 0.058 | -0.007 | 0.019 |
| VideoMAE-large              | pretrained | 0.156  | 0.136    | 0.076  | 0.152  | 0.197 | 0.176  | 0.171 |
| VideoMAE-large              | random     | 0.011  | 0.013    | 0.064  | -0.015 | -0.031| 0.067  | -0.003|
| DINOv2-large (image)        | pretrained | 0.122  | 0.100    | 0.078  | 0.141  | 0.170 | 0.137  | 0.007 |
| DINOv2-large (image)        | random     | 0.092  | 0.077    | 0.015  | 0.093  | 0.149 | 0.092  | 0.073 |

Per-family Δr (pretrained − random):

| Family               | Δr mean | Δr median |
|----------------------|---------|-----------|
| V-JEPA-2 ViT-L       | +0.050  | +0.032    |
| VideoMAE-large       | **+0.145** | **+0.123** |
| DINOv2-large (image) | +0.030  | +0.023    |

**Read.** If VideoMAE Δr ≈ V-JEPA-2 Δr, then masked-feature vs
masked-pixel doesn't matter for mouse-VIS predictivity at this scale —
the JEPA-vs-pixel-MAE distinction is washed out by ridge. If VideoMAE
Δr ≪ V-JEPA-2 Δr, latent-prediction objective genuinely buys
predictivity. If VideoMAE Δr > V-JEPA-2 Δr, pixel-MAE is the better
mouse-VIS encoder and the framing of this whole project flips.


## Stage 4 — Hénaff straightening, mean per-step trajectory curvature θ

Reference rows (constants for this session/stim):

| Trajectory                        | θ      |
|-----------------------------------|--------|
| Pixels (304×608, 184832-D)        | 84.5°  |
| Mouse VIS pop. (avg 20 repeats)   | 115.8° |
| i.i.d. baseline (consecutive Δ)   | 120°   |

Models:

| Model           | Init       | θ      | Δ vs pixels | \|θ − cortex\| |
|-----------------|------------|--------|-------------|----------------|
| V-JEPA-2 ViT-L  | pretrained | 160.2° | +75.7°      | 44.4°          |
| V-JEPA-2 ViT-L  | random     | 96.6°  | +12.1°      | 19.2°          |
| VideoMAE-large  | pretrained | 109.8° | +25.3°      | **6.0°**       |
| VideoMAE-large  | random     | 80.4°  | -4.1°       | 35.4°          |
| DINOv2-large    | pretrained | 101.5° | +17.0°      | 14.3°          |
| DINOv2-large    | random     | 85.1°  | +0.6°       | 30.7°          |

Per-family curving learned by training (θ_pretrained − θ_random):

| Family               | Δθ (learned curving) |
|----------------------|----------------------|
| V-JEPA-2 ViT-L       | +63.6°               |
| VideoMAE-large       | +29.4°               |
| DINOv2-large         | +16.4°               |

**Read.** V-JEPA-2 pretrained sits 44° *more curved* than mouse VIS;
random init sits 19° below. If VideoMAE pretrained also overshoots cortex
by ~40°+, the +64° curving is generic to deep ViT + masked objective and
not JEPA-specific. If VideoMAE pretrained sits closer to cortex (small
|θ − cortex|), pixel-MAE has the more cortex-like geometry despite
predicting worse, and the Stage 3 / Stage 4 dissociation noted on
2026-05-02 (predictive ≠ geometrically isomorphic) sharpens further.


## Status

- [x] Run 1 — Stage 3, VideoMAE pretrained → `lewm-brain-stage3-vmae` (2026-05-03)
- [x] Run 2 — Stage 3, VideoMAE random → `lewm-brain-stage3-vmae-rand` (2026-05-03)
- [x] Run 3 — Stage 4, VideoMAE pretrained → `lewm-brain-stage4-vmae` (2026-05-03)
- [x] Run 4 — Stage 4, VideoMAE random → `lewm-brain-stage4-vmae-rand` (2026-05-03)
- [x] Run 5 — Stage 2, DINOv2 pretrained → `lewm-brain-stage2-dino-l16` (2026-05-03)
- [x] Run 6 — Stage 2, DINOv2 random → `lewm-brain-stage2-dino-l16-rand` (2026-05-03)
- [x] Run 7 — Stage 3, DINOv2 pretrained → `lewm-brain-stage3-dino` (2026-05-03)
- [x] Run 8 — Stage 3, DINOv2 random → `lewm-brain-stage3-dino-rand` (2026-05-03)
- [x] Run 9 — Stage 4, DINOv2 pretrained → `lewm-brain-stage4-dino` (2026-05-03)
- [x] Run 10 — Stage 4, DINOv2 random → `lewm-brain-stage4-dino-rand` (2026-05-03)

Kaggle gotcha (carry over from 2026-05-03): re-run Cell 2 after
`git pull`, and append `--force-reinstall --no-deps` to the lewm-brain
pip line in Cell 1, or stale-wheel + stale-config will bite again.
