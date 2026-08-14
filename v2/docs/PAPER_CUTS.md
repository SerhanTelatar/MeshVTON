# Material cut from the paper for the page limit

The dissertation paper had to come down from 12 pages to 8. Nothing here was cut because it
was wrong — it was cut because it was repeated elsewhere, or because a figure duplicated a
table. This file keeps it for the video, the slides and the viva, where there is no page limit
and where several of these points are the most interesting things to say out loud.

One exception is flagged at the bottom: a claim that was removed because it was **never
measured**. Do not put it back.

---

## 1. Why this problem is harder than 2D try-on (cut from the Introduction)

Three specific difficulties, none of which a 2D try-on paper faces. Good opening material for
a talk, because it explains why the results look the way they do.

1. **No paired supervision exists.** No dataset provides a 3D garment mesh together with a real
   photograph of the same person wearing it. Supervision had to be assembled from a synthetic
   pipeline and a real dataset that share no common ground truth.
2. **The conditioning is estimated, not given.** It is produced in screen space from a body and
   camera that are themselves regressed from a single image, so any error in that estimate
   propagates into every downstream measurement.
3. **The backbone is frozen.** A 12B transformer with 1.7% trainable parameters cannot simply be
   fitted to the task; the geometric signal has to be injected without disturbing the
   pretrained prior.

## 2. Objectives, stated separately from the research questions (cut from the Introduction)

- **O1** Build a conditioning pipeline that produces identical inputs for training and inference.
- **O2** Define and implement a measure of mesh specificity together with an untrained null control.
- **O3** Calibrate the conditioning alignment against an independent reference, without training.
- **O4** Ablate the geometry branch to separate its contribution.
- **O5** Ablate inference-time controls to localise the appearance limitation.

Every one of these was completed. If a slide needs a "what did you set out to do / did you do
it" structure, this is it.

## 3. The three failure modes the design had to survive (cut from the Introduction)

- **Viewpoint mismatch.** If the rendered geometry does not match the pose in the reference
  image the result degrades. Fixed by building training and inference conditioning from a
  single shared pipeline, so the view always matches the target camera.
- **Conditioning misalignment.** The garment mesh can hang too high or be scaled too small
  relative to the estimated body, which silently degrades every downstream metric. Fixed by
  the training-free calibration.
- **Appearance.** Even with a fully textured reference the model does not copy colour and
  texture well. Localised to the reference-based appearance path, not the geometry.

## 4. Positioning table (cut: was Table I)

| Method | Diffusion prior | 3D geometry | Frozen backbone | Multi-view |
|---|---|---|---|---|
| VITON-HD | no | no | no | no |
| IDM-VTON | yes | no | no | no |
| CatVTON | yes | no | yes | no |
| M3D-VTON | no | yes | no | yes |
| **MeshVTON (ours)** | yes | yes | yes | yes |

This is a good slide. It was cut from the paper because one sentence carried the same claim,
but as a visual it lands better than the sentence does.

## 5. Self-referential metrics (cut as a Discussion subsection, survives as one paragraph)

The full argument, which is worth telling slowly in a talk because it is the most transferable
lesson in the project:

> Our initial evaluation used silhouette overlap alone. When we corrected the conditioning
> misalignment, that metric **decreased**, even though the outputs improved visibly and by every
> independent measure. The reason is structural: the metric compares the output with the
> conditioning we supplied, so when the conditioning moves, the target moves with it. This is
> worth stating plainly because agreement-with-conditioning metrics are common in the
> controllable-generation literature. A cross-matched specificity test, or an agreement measure
> referenced to an external parser, is needed to detect a systematically wrong conditioning
> signal.

## 6. How the specificity test relates to existing controls (cut from Related Work)

> Related controls do exist. In virtual try-on, garment-agnostic evaluation withholds the
> garment reference entirely to test whether it is used at all, which is analogous to the on/off
> ablation we also report. Our specificity test extends this idea to a cross-matched form:
> rather than removing the condition, we score each output against the targets of *other*
> garments and report the diagonal-minus-off-diagonal gap as a single number. This gives a
> scalar that is zero for a model ignoring the specific mesh, and it can be computed with no
> extra generation from outputs already produced for evaluation.

## 7. Figures cut (the PNGs still exist under `v2/eval_results/figures/`)

| Figure | Why it was cut | Use in the video? |
|---|---|---|
| `fig_ablation` (control off vs on) | subsumed by `fig_comparison` | yes — cleanest single before/after |
| `fig_qual` (reference vs output) | subsumed by `fig_comparison` | maybe |
| `fig_multiview` (one person, several viewpoints) | `fig_comparison` already shows four viewpoints | yes — best "it follows the pose" shot |
| `fig_calibration` (hang and scale sweeps) | Table V has the same three numbers | **yes** — the two curves with a clear interior optimum are far more convincing than the table |

`fig_calibration` in particular should go in the video. A sweep with a single clear peak, far
from the value inherited from the synthetic pipeline, is the strongest visual argument that the
calibration finding is real and not a fluke.

## 8. Future work items cut

- **Per-person calibration.** The optimal hang offset varied across individuals from $-0.21$ to
  $0.00$ metres. A per-person estimate, derived from the parser garment region at inference
  time, would recover the spread that a single global constant cannot. (Cut because
  limitation (v) already states the same range.)
- **Training budget and backbone choice.** The control branch and adapter were trained for 4000
  steps on a 12B backbone. A smaller backbone with pretrained geometric ControlNets would allow
  an order of magnitude more optimisation steps for the same compute. (Cut because it is the
  one suggestion with no measurement behind it — and the record argues against it, see
  [[meshvton-v2-plan]]: a completed 4000-step run was also washed out.)

## 9. Longer ethics discussion (cut to half length)

The removed detail: a deployed system requires explicit consent for the upload **and
processing** of a likeness, not merely for upload; and the misrepresentation risk was spelled
out as "such an output must not be presented as a product photograph", which is the concrete
rule a deployment would need to enforce.

## 10. Supporting sentence cut from the calibration section

> We note that the uncalibrated configuration closely reproduces the silhouette overlap of
> $0.504$ we obtained in earlier work with a different garment selection, which indicates that
> the improvement is attributable to the calibration rather than to the choice of evaluation
> garments.

Cut because it points at unpublished earlier work a reader cannot check. It is still a fair
answer if someone in the viva asks "how do you know the gain is not just an easier garment set?"

## 11. Long-form captions

The captions were reduced to bare identification. The interpretation they used to carry is all
still in the body text — but the caption wording is often the better phrasing for a slide.
Notably: *"the baseline transfers colour and pattern faithfully but leaves the original garment
silhouette unchanged, whereas MeshVTON adopts the off-shoulder cut of the reference mesh while
losing colour fidelity"* — that one sentence is the whole result.

---

## DO NOT REUSE: the claim that was removed for being unmeasured

The paper used to state, in two places:

> Outside the mask the outputs stay almost identical to the input person (SSIM $>0.998$)

and

> the baseline leaves the surrounding region bit-identical (SSIM $=1.000$, LPIPS $=0.000$)

**These numbers have no source.** `eval_checkpoint.py` writes `.ref/.mask/.sil/.predsil` but not
`.person.png`, and `harness.py::evaluate_item` only computes `ssim_outside` / `lpips_outside`
when a `.person` aux file is present. The report for the July checkpoint contains only
`garment_delta_e`, `geo_iou` and `silhouette_iou` — confirmed 2026-08-13.

The claim itself is true by construction (generated content is composited back inside the
inpainting mask only, so everything outside is carried over unchanged), and that is how the
paper now words it. Say it that way in the video too. Do not quote a number.

To make the number real, `eval_checkpoint.py` would have to save the person image alongside
each prediction, and the evaluation would have to be re-run.
