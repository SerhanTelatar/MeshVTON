# Two training regimes — which one, why, and how to bring it back

This repository is a **single code base**; it produced two different training regimes.
The code was never forked, the difference is only in the data and the flags. The thesis
results rest on **regime A**.

## A — TEXTURED (the regime the thesis rests on)

- Checkpoint: `<Drive>/v2_outputs/stage1_july/ckpt_004000.pt` (2026-07-06, 4000 steps)
- Appearance ref: **textured** (the garment's real color/pattern)
- Training data: 70% real VITON-HD + 30% synthetic multi-view
- Golden set: textured garments (`manifest.JULY.json`)
- Inference: `--use-texture`, `hang_pad=-0.12`, `garment_scale=1.25`

Result (n=50): geo_iou 0.5234 → **0.7005**, mesh specificity +0.0778 → **+0.2205**,
the control branch's share **65%**. The untrained baseline's specificity is +0.0046 (≈ zero).
The outputs are photorealistic and colored; limitation: garment *identity* is tracked loosely.

## B — TEXTURELESS (the 2026-08 regime)

- Checkpoint: `<Drive>/v2_outputs/stage1/final.pt` (2026-08-10, 1000 steps)
- Appearance ref: **always flat grey** via `force_textureless`
- Training data: synthetic only (VITON-HD deliberately removed)
- Golden set: `manifest.AUG.json`

On the same set: geo_iou 0.6072, specificity +0.1368, control share 41%.
**Below A on every metric.** The hypothesis "removing the appearance path strengthens the
geometry" was refuted by this measurement. The outputs are grey and flat.

## The points in the code that make the difference

| topic | A (textured) | B (textureless) | where |
|---|---|---|---|
| appearance ref | `use_texture=True` | default `False` | `builder.py::build_conditioning` |
| synthetic garment selection | textured only | all | `synth/generate.py::discover_assets` |
| real data | VITON-HD mixed in | none | `colab_pipeline.py` (vitonhd stage removed) |
| golden set | textured garments | any | `build_golden_set.py --garment-ids` |

The `use_texture` default stays **False** (project rule). Running regime A requires passing
the flag explicitly — `eval_checkpoint.py --use-texture`,
`MeshVTON_inference_stage1.ipynb`.

## If retraining is needed (regime A)

No code fork is required, three changes are enough:

1. Run `generate_synthetic.py` with textured garments (bring the texture filter back) and
   pass `build_conditioning(..., use_texture=True)` — the GT and the ref must have the same
   look, otherwise the flow-matching loss collapses to the conditional mean (the 2026-08-09
   lesson).
2. Add the VITON-HD stage back to `colab_pipeline.py` (`preprocess_vitonhd.py` is still
   there as a script).
3. Bump `DATA_VERSION` — the synthetic data will be regenerated (~3 hours).

Cost: ~3 h of data + ~11 h of training (4000 steps, 10 s/step on an A100).

## Things that must not be deleted

- `stage1_july/ckpt_004000.pt` — the ONLY checkpoint the thesis rests on. It was rescued
  from the trash; do NOT put it under `stage1/`, training rotation (`keep_last=2`) deletes it.
- `manifest.JULY.json` and `manifest.AUG.json` — the golden sets of the two regimes.
- `eval_results/` — all measurements and figures.
