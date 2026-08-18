#!/usr/bin/env python3
"""Builds the CASE figure and a failure taxonomy from an existing eval report.

Reporting what does not work is part of the assessment, and a limitations section carries far
more weight when the failures are selected by measurement rather than by eye. This script ranks
every evaluated combination by the metric that defines each failure mode, takes the extreme
cases, and lays them out with their measured values printed on the panel.

--rank selects which end of each ranking is shown. The metrics span a wide range (see the
report), so showing only the worst end misrepresents the method: --rank both puts the best and
the worst cases of the same metric in one figure and makes the spread visible.

NO diffusion, NO GPU — it reads the report JSON and the PNGs already written by
eval_checkpoint.py.

Failure modes, each keyed to the metric that detects it:
  colour     — garment_delta_e high: colour/pattern not transferred (our main limitation)
  placement  — geo_iou low: the garment does not follow the supplied silhouette
  specificity— per-person specificity low: the output barely depends on which mesh was asked for
               (requires --specificity-json from mesh_specificity.py --per-person)

Usage:
  python v2/scripts/failure_cases.py --report v2/eval_results/ckpt_control_on_july.json
  [--set ckpt_control_on_july] [--top 4] [--thumb 220] [--rank both]
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
    # key: (metric, worst = highest?, worst-end title, best-end title)
    "colour": ("garment_delta_e", True,
               "colour not transferred (high dE)",
               "colour transferred (low dE)"),
    "placement": ("geo_iou", False,
                  "garment does not follow the silhouette (low geo_IoU)",
                  "garment follows the silhouette (high geo_IoU)"),
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
    ap.add_argument("--top", type=int, default=4, help="N cases per block")
    ap.add_argument("--rank", choices=("worst", "best", "both"), default="worst",
                    help="which end of each ranking to show; 'both' adds a best-case block "
                         "above every worst-case block")
    ap.add_argument("--per-row", type=int, default=2,
                    help="cases per row inside a block: 4 = wide (\\begin{figure*}), "
                         "2 = roughly square, 1 = tall single column")
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

    # ---- pick the extreme cases per mode, by measurement ----
    ends = ("best", "worst") if args.rank == "both" else (args.rank,)
    picks: list[tuple[str, str, list[dict]]] = []
    for key, (metric, worst_high, worst_title, best_title) in MODES.items():
        vals = [r for r in rows if isinstance(r.get(metric), (int, float))]
        if not vals:
            print(f"SKIP {key}: '{metric}' is missing from the report", file=sys.stderr)
            continue
        # worst_high says which direction is the failure, so the best end is the opposite sort
        vals.sort(key=lambda r: r[metric], reverse=worst_high)
        for end in ends:
            ranked = vals if end == "worst" else vals[::-1]
            title = f"{end}: " + (worst_title if end == "worst" else best_title)
            picks.append((key, title, ranked[: args.top]))
        arr = np.array([r[metric] for r in vals], float)
        print(f"{key:11s} {metric:18s} n={len(arr)} mean={arr.mean():.3f} "
              f"worst={arr[0] if worst_high else arr.min():.3f} "
              f"best={arr.min() if worst_high else arr.max():.3f}")

    if not picks:
        raise SystemExit("ERROR: no failure mode could be built")

    # The person photo comes from the `.person.png` written next to every prediction, NOT from
    # the golden manifest: eval writes it (harness.py::_aux_paths) and it is exactly the image
    # that prediction was produced from, whereas v2/data/golden/ is gitignored and is often
    # absent on a machine that only has the restored eval_results.
    def person_path(pid: str, gid: str) -> Path | None:
        p = Path(f"{stem(pid, gid)}.person.png")
        if p.exists():
            return p
        g = by_pid.get(pid)
        q = (manifest.root / g.image) if g else None
        return q if q is not None and q.exists() else None

    all_cases = [r for _, _, cs in picks for r in cs]
    person_ok = any(person_path(r["person"], r["garment"]) for r in all_cases)
    cols = (["person"] if person_ok else []) + ["reference", "output"]
    if not person_ok:
        print("NOTE: neither the .person.png aux files nor the golden-set photographs were "
              "found — the input column is omitted.", file=sys.stderr)

    def panels_for(pid: str, gid: str) -> list[Image.Image | None]:
        out = [load(Path(f"{stem(pid, gid)}.ref.png"), sz), load(Path(f"{stem(pid, gid)}.png"), sz)]
        if person_ok:
            pp = person_path(pid, gid)
            out.insert(0, load(pp, sz) if pp else None)
        return out

    def put(d: ImageDraw.ImageDraw, im: Image.Image, img: Image.Image | None, x: int, y: int):
        if img is not None:
            im.paste(img, (x, y))
        else:
            d.rectangle([x, y, x + TH - 2, y + hh - 2], fill=(235, 235, 235))

    # One block per failure mode; inside a block the cases are tiled --per-row across, so the
    # aspect ratio is chosen by the caller: 4 = short and wide (\begin{figure*}), 2 = roughly
    # square, 1 = one case per row (tall, single column).
    HEAD, LABEL, FOOT = 26, 22, 20
    NC = len(cols)
    PR = max(1, min(args.per_row, args.top))
    case_rows = -(-args.top // PR)  # ceil
    block_h = HEAD + case_rows * (hh + LABEL)

    im = Image.new("RGB", (TH * NC * PR, block_h * len(picks) + FOOT), "white")
    d = ImageDraw.Draw(im)
    for b, (key, title, cases) in enumerate(picks):
        y0 = b * block_h
        d.rectangle([0, y0, im.width, y0 + HEAD - 2], fill=(240, 240, 240))
        d.text((6, y0 + 7), f"{key}: {title}", fill="black")
        for i, r in enumerate(cases):
            x0 = (i % PR) * TH * NC
            y = y0 + HEAD + (i // PR) * (hh + LABEL)
            for c, img in enumerate(panels_for(r["person"], r["garment"])):
                put(d, im, img, x0 + c * TH, y)
            metric = MODES[key][0]
            d.text((x0 + 4, y + hh + 4),
                   f"{r['person']} x {r['garment'].split('__')[-1]}  "
                   f"{metric}={r[metric]:.2f}", fill="black")
            if i % PR:  # separator between cases on the same row
                d.line([x0, y, x0, y + hh], fill=(200, 200, 200), width=2)

    d.text((4, im.height - FOOT + 4), f"each case: {' | '.join(cols)}", fill="black")
    suffix = "" if args.rank == "worst" else f"__{args.rank}"
    out = fig_dir / f"fig_failures__{pred_set}{suffix}.png"
    im.save(out)
    print(f"\n{out}")
    print(f"Selected BY MEASUREMENT ({args.rank} per metric), not by eye — say so in the caption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
