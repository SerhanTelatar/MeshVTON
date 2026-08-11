"""Eval harness: golden manifest + prediction directory -> metric table (JSON + markdown).

Prediction contract (same as configs/eval.yaml):
  pred_dir/{person_id}_{garment_id}_{angle:03d}.png
Optional auxiliary files with the same stem:
  {stem}.person.png  {stem}.mask.png  {stem}.sil.png  {stem}.ref.png  {stem}.gt.png
Each metric is computed only if its required inputs exist; missing ones are reported as "n/a".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from meshvton2.eval import metrics as M
from meshvton2.eval.golden_set import GoldenManifest

PRED_PATTERN = "{person_id}_{garment_id}_{angle:03d}.png"
AUX_SUFFIXES = ("person", "mask", "sil", "ref", "gt", "predsil")
# predsil = the garment mask the parser extracts from the GENERATED image — geo_iou is measured
# with it (comparing the conditioning silhouette against the conditioning is constant in an ablation)


@dataclass
class EvalRow:
    person_id: str
    garment_id: str
    angle: int
    found: bool
    values: dict  # metric name -> float | None


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != (size[1], size[0]):
        img = img.resize((size[1], size[0]), Image.LANCZOS)
    return np.asarray(img)


def _load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None and img.size != (size[1], size[0]):
        img = img.resize((size[1], size[0]), Image.NEAREST)
    return np.asarray(img) > 127


def _aux_paths(pred_path: Path) -> dict[str, Path]:
    stem = pred_path.with_suffix("")
    return {s: Path(f"{stem}.{s}.png") for s in AUX_SUFFIXES}


def evaluate_item(pred_path: Path) -> dict:
    """Every metric computable for a single prediction with the inputs available."""
    pred = _load_rgb(pred_path)
    size = pred.shape[:2]
    aux = {k: p for k, p in _aux_paths(pred_path).items() if p.exists()}
    values: dict[str, float | None] = {}

    if "person" in aux and "mask" in aux:
        person = _load_rgb(aux["person"], size)
        mask = _load_mask(aux["mask"], size)
        values.update(M.outside_mask_preservation(pred, person, mask))

    if "ref" in aux and "mask" in aux:
        ref = _load_rgb(aux["ref"])
        mask = _load_mask(aux["mask"], size)
        values["garment_delta_e"] = M.garment_delta_e(pred, mask, ref)

    if "sil" in aux and "mask" in aux:
        values["silhouette_iou"] = M.silhouette_iou(
            _load_mask(aux["mask"], size), _load_mask(aux["sil"], size)
        )

    if "predsil" in aux and "sil" in aux:
        # geometry fidelity: does the garment region IN THE OUTPUT match the target silhouette
        values["geo_iou"] = M.silhouette_iou(
            _load_mask(aux["predsil"], size), _load_mask(aux["sil"], size)
        )

    if "gt" in aux:
        gt = _load_rgb(aux["gt"], size)
        values["gt_ssim"] = M.compute_ssim(pred, gt)
        values["gt_psnr"] = M.compute_psnr(pred, gt)
        values["gt_lpips"] = M.compute_lpips(pred, gt)
        if "mask" in aux:
            mask = _load_mask(aux["mask"], size)
            values["gt_garment_ssim"] = M.compute_ssim(pred, gt, mask=mask)
            values["gt_garment_lpips"] = M.compute_lpips(pred, gt, mask=mask)

    return values


def evaluate(manifest: GoldenManifest, pred_dir: str | Path, pred_pattern: str = PRED_PATTERN) -> dict:
    pred_dir = Path(pred_dir)
    rows: list[EvalRow] = []
    for person, garment, angle in manifest.items():
        pred_path = pred_dir / pred_pattern.format(
            person_id=person.id, garment_id=garment.id, angle=angle
        )
        if not pred_path.exists():
            rows.append(EvalRow(person.id, garment.id, angle, found=False, values={}))
            continue
        rows.append(EvalRow(person.id, garment.id, angle, found=True, values=evaluate_item(pred_path)))
    return summarize(rows)


def summarize(rows: list[EvalRow]) -> dict:
    metric_names = sorted({k for r in rows for k in r.values})

    def agg(subset: list[EvalRow]) -> dict:
        out = {}
        for name in metric_names:
            vals = [r.values[name] for r in subset if r.values.get(name) is not None and np.isfinite(r.values[name])]
            out[name] = {"mean": float(np.mean(vals)), "n": len(vals)} if vals else {"mean": None, "n": 0}
        return out

    found = [r for r in rows if r.found]
    per_angle = {}
    for angle in sorted({r.angle for r in rows}):
        per_angle[str(angle)] = agg([r for r in found if r.angle == angle])
    return {
        "total": len(rows),
        "found": len(found),
        "missing": [f"{r.person_id}_{r.garment_id}_{r.angle:03d}" for r in rows if not r.found],
        "overall": agg(found),
        "per_angle": per_angle,
        "rows": [
            {"person": r.person_id, "garment": r.garment_id, "angle": r.angle, **r.values}
            for r in found
        ],
    }


def write_report(summary: dict, out_json: str | Path, out_md: str | Path | None = None) -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if out_md is None:
        out_md = out_json.with_suffix(".md")

    def fmt(cell) -> str:
        if cell is None or cell["mean"] is None:
            return "n/a"
        return f"{cell['mean']:.4f} (n={cell['n']})"

    names = sorted(summary["overall"].keys())
    lines = [
        f"# Eval report",
        f"- Predictions: {summary['found']}/{summary['total']} found"
        + (f", missing: {len(summary['missing'])}" if summary["missing"] else ""),
        "",
        "| metric | overall | " + " | ".join(f"{a}°" for a in summary["per_angle"]) + " |",
        "|---|---|" + "---|" * len(summary["per_angle"]),
    ]
    for name in names:
        row = [fmt(summary["overall"][name])] + [fmt(summary["per_angle"][a].get(name)) for a in summary["per_angle"]]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    Path(out_md).write_text("\n".join(lines) + "\n")
