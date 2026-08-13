#!/usr/bin/env python3
"""Builds the FAILURE-CASE figure and a failure taxonomy from an existing eval report.

Reporting what does not work is part of the assessment, and a limitations section carries far
more weight when the failures are selected by measurement rather than by eye. This script ranks
every evaluated combination by the metric that defines each failure mode, takes the worst cases,
and lays them out with their measured values printed on the panel.

NO diffusion, NO GPU — it reads the report JSON and the PNGs already written by
eval_checkpoint.py.

Failure modes, each keyed to the metric that detects it:
  colour     — garment_delta_e high: colour/pattern not transferred (our main limitation)
  placement  — geo_iou low: the garment does not follow the supplied silhouette
  specificity— per-person specificity low: the output barely depends on which mesh was asked for
               (requires --specificity-json from mesh_specificity.py --per-person)

Usage:
  python v2/scripts/failure_cases.py --report v2/eval_results/ckpt_control_on_july.json
  [--set ckpt_control_on_july] [--top 4] [--thumb 220]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from meshvton2.eval.golden_set import load_manifest  # noqa: E402

MODES = {
    # key: (metric, worst = highest?, panel title)
    "colour": ("garment_delta_e", True, "colour not transferred (high dE)"),
    "placement": ("geo_iou", False, "garment does not follow the silhouette (low geo_IoU)"),
}


def load(p: Path, size: tuple[int, int]) -> Image.Image | None:
    if not p.exists():
        return None
    return Image.open(p).convert("RGB").resize(size[::-1], Image.LANCZOS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--set", dest="pred_set", default=None,
                    help="preds folder under eval_results; default: the report file name")
    ap.add_argument("--top", type=int, default=4, help="worst N cases per failure mode")
    ap.add_argument("--layout", choices=("tall", "wide"), default="wide",
                    help="wide = cases side by side (use \\begin{figure*}); tall = one per row")
    ap.add_argument("--thumb", type=int, default=220)
    ap.add_argument("--angle", type=int, default=0)
    args = ap.parse_args()

    if not args.report.is_file():
        raise SystemExit(f"ERROR: no report: {args.report}")
    rep = json.loads(args.report.read_text())
    rows = [r for r in rep.get("rows", []) if r.get("angle") == args.angle]
    if not rows:
        raise SystemExit("ERROR: the report has no row at this angle")

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    out_root = REPO / base["paths"]["eval_results"]
    pred_set = args.pred_set or args.report.stem
    pred_dir = out_root / pred_set / "preds"
    if not pred_dir.is_dir():
        raise SystemExit(f"ERROR: no preds folder: {pred_dir}")
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_pid = {p.id: p for p in manifest.persons}
    TH = args.thumb
    hh = round(TH * base["resolution"]["height"] / base["resolution"]["width"])
    sz = (hh, TH)
    stem = lambda pid, gid: (pred_dir / cfg["pred_pattern"].format(
        person_id=pid, garment_id=gid, angle=args.angle)).with_suffix("")

    # ---- pick the worst cases per mode, by measurement ----
    picks: list[tuple[str, str, list[dict]]] = []
    for key, (metric, worst_high, title) in MODES.items():
        vals = [r for r in rows if isinstance(r.get(metric), (int, float))]
        if not vals:
            print(f"SKIP {key}: '{metric}' is missing from the report", file=sys.stderr)
            continue
        vals.sort(key=lambda r: r[metric], reverse=worst_high)
        picks.append((key, title, vals[: args.top]))
        arr = np.array([r[metric] for r in vals], float)
        print(f"{key:11s} {metric:18s} n={len(arr)} mean={arr.mean():.3f} "
              f"worst={arr[0] if worst_high else arr.min():.3f} "
              f"best={arr.min() if worst_high else arr.max():.3f}")

    if not picks:
        raise SystemExit("ERROR: no failure mode could be built")

    # The golden-set person photographs are gitignored, so on a fresh machine they may be absent.
    # A blank column looks like a broken figure, so drop it rather than drawing grey boxes.
    all_cases = [r for _, _, cs in picks for r in cs]
    person_ok = any((p := by_pid.get(r["person"])) is not None
                    and (manifest.root / p.image).exists() for r in all_cases)
    cols = (["person"] if person_ok else []) + ["reference", "output"]
    if not person_ok:
        print("NOTE: golden-set person photographs not found — the input column is omitted "
              "(the output column already shows the person).", file=sys.stderr)

    def panels_for(pid: str, gid: str) -> list[Image.Image | None]:
        out = [load(Path(f"{stem(pid, gid)}.ref.png"), sz), load(Path(f"{stem(pid, gid)}.png"), sz)]
        if person_ok:
            person = by_pid.get(pid)
            out.insert(0, load(manifest.root / person.image, sz) if person else None)
        return out

    def put(d: ImageDraw.ImageDraw, im: Image.Image, img: Image.Image | None, x: int, y: int):
        if img is not None:
            im.paste(img, (x, y))
        else:
            d.rectangle([x, y, x + TH - 2, y + hh - 2], fill=(235, 235, 235))

    HEAD, LABEL, FOOT = 26, 22, 20
    NC = len(cols)

    if args.layout == "wide":
        # Cases run ACROSS, each case occupying NC panels. One row per failure mode, so the
        # figure is short and wide -- the shape a two-column \begin{figure*} wants.
        block_h = HEAD + hh + LABEL
        im = Image.new("RGB", (TH * NC * args.top, block_h * len(picks) + FOOT), "white")
        d = ImageDraw.Draw(im)
        for b, (key, title, cases) in enumerate(picks):
            y0 = b * block_h
            d.rectangle([0, y0, im.width, y0 + HEAD - 2], fill=(240, 240, 240))
            d.text((6, y0 + 7), f"{key}: {title}", fill="black")
            for i, r in enumerate(cases):
                x0 = i * TH * NC
                for c, img in enumerate(panels_for(r["person"], r["garment"])):
                    put(d, im, img, x0 + c * TH, y0 + HEAD)
                metric = MODES[key][0]
                d.text((x0 + 4, y0 + HEAD + hh + 4),
                       f"{r['person']} x {r['garment'].split('__')[-1]}  "
                       f"{metric}={r[metric]:.2f}", fill="black")
                if i:  # separator between cases
                    d.line([x0, y0 + HEAD, x0, y0 + HEAD + hh], fill=(200, 200, 200), width=2)
    else:
        block_h = HEAD + args.top * (hh + LABEL)
        im = Image.new("RGB", (TH * NC, block_h * len(picks) + FOOT), "white")
        d = ImageDraw.Draw(im)
        for b, (key, title, cases) in enumerate(picks):
            y0 = b * block_h
            d.rectangle([0, y0, im.width, y0 + HEAD - 2], fill=(240, 240, 240))
            d.text((6, y0 + 7), f"{key}: {title}", fill="black")
            for i, r in enumerate(cases):
                y = y0 + HEAD + i * (hh + LABEL)
                for c, img in enumerate(panels_for(r["person"], r["garment"])):
                    put(d, im, img, c * TH, y)
                metric = MODES[key][0]
                d.text((4, y + hh + 4),
                       f"{r['person']} x {r['garment'].split('__')[-1]}  "
                       f"{metric}={r[metric]:.2f}", fill="black")

    d.text((4, im.height - FOOT + 4),
           f"each case: {' | '.join(cols)}" if args.layout == "wide"
           else "columns: " + " | ".join(cols), fill="black")
    out = fig_dir / f"fig_failures__{pred_set}.png"
    im.save(out)
    print(f"\n{out}")
    print("Selected BY MEASUREMENT (worst per metric), not by eye — say so in the caption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
