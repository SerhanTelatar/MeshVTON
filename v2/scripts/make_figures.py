#!/usr/bin/env python3
"""Builds the thesis figures from the existing predictions — NO DIFFUSION, it only reads files.

It produces two figures:

1. `fig_<set>_persons.png` — row = person, columns = [person photo, the output for each garment].
   The "same person, different mesh" figure; the visual counterpart of mesh specificity.

2. `fig_<set>_refs.png` — row = combo, columns = [person, garment reference, output].
   The Fig 4 layout from the report. The reference is already written as `.ref.png` during eval.

If `--sets A B` is given, the two runs are laid out SIDE BY SIDE in the second column group
(e.g. the textured July run vs the textureless August one) — the checkpoint comparison figure.

Usage:
  python v2/scripts/make_figures.py --sets ckpt_control_on_july
  python v2/scripts/make_figures.py --sets ckpt_control_on_july ckpt_control_on_aug
  [--limit-persons 5] [--thumb 220]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from meshvton2.eval.golden_set import load_manifest  # noqa: E402


def load(p: Path, size: tuple[int, int]) -> np.ndarray | None:
    if not p.exists():
        return None
    return np.asarray(Image.open(p).convert("RGB").resize(size[::-1], Image.LANCZOS))


def label_strip(width: int, texts: list[str], cell: int, h: int = 26) -> Image.Image:
    """Column headers — so the figure can be read on its own."""
    im = Image.new("RGB", (width, h), "white")
    d = ImageDraw.Draw(im)
    for i, t in enumerate(texts):
        d.text((i * cell + 6, 7), t[:28], fill="black")
    return im


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", required=True, help="pred folders under eval_results")
    ap.add_argument("--limit-persons", type=int, default=5)
    ap.add_argument("--limit-combos", type=int, default=4,
                    help="rows in figure 2; the default keeps it short enough for one column")
    ap.add_argument("--thumb", type=int, default=220)
    ap.add_argument("--angle", type=int, default=0)
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    out_root = REPO / base["paths"]["eval_results"]
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    TH = args.thumb
    hh = round(TH * base["resolution"]["height"] / base["resolution"]["width"])
    sz = (hh, TH)

    by_person: dict[str, list[str]] = {}
    for c in manifest.combos:
        by_person.setdefault(c.person_id, [])
        if c.garment_id not in by_person[c.person_id]:
            by_person[c.person_id].append(c.garment_id)
    persons = list(by_person)[: args.limit_persons]
    by_pid = {p.id: p for p in manifest.persons}
    stem = lambda s, pid, gid: (out_root / s / "preds" / cfg["pred_pattern"].format(
        person_id=pid, garment_id=gid, angle=args.angle)).with_suffix("")

    missing = [s for s in args.sets if not (out_root / s / "preds").exists()]
    if missing:
        raise SystemExit(f"ERROR: no preds folder: {missing}")

    # --- Figure 1: row = person, columns = person + garments (one block per set) ---
    gids = by_person[persons[0]]
    cols = 1 + len(gids) * len(args.sets)
    grid = Image.new("RGB", (TH * cols, hh * len(persons)), "white")
    for r, pid in enumerate(persons):
        person_img = load(manifest.root / by_pid[pid].image, sz)
        if person_img is not None:
            grid.paste(Image.fromarray(person_img), (0, r * hh))
        c = 1
        for s in args.sets:
            for gid in gids:
                a = load(Path(f"{stem(s, pid, gid)}.png"), sz)
                if a is not None:
                    grid.paste(Image.fromarray(a), (c * TH, r * hh))
                c += 1
    heads = ["person"] + [f"{s.replace('ckpt_control_on', '')}|{g.split('__')[-1]}"
                          for s in args.sets for g in gids]
    f1 = Image.new("RGB", (grid.width, grid.height + 26), "white")
    f1.paste(label_strip(grid.width, heads, TH), (0, 0))
    f1.paste(grid, (0, 26))
    p1 = fig_dir / f"fig_persons__{'_vs_'.join(args.sets)}.png"
    f1.save(p1)
    print(f"[1] {p1}  ({len(persons)} people × {len(gids)} garments × {len(args.sets)} sets)")

    # --- Figure 2: row = combo, columns = person | reference | output(s) ---
    # One combo per person first, so the rows show DIFFERENT people rather than one person
    # repeated with every garment — and cap the count, a 15-row figure does not fit a column.
    rows = [(pid, gids[i % len(gids)]) for i, pid in enumerate(persons)][: args.limit_combos]
    cols2 = 2 + len(args.sets)
    grid2 = Image.new("RGB", (TH * cols2, hh * len(rows)), "white")
    for r, (pid, gid) in enumerate(rows):
        person_img = load(manifest.root / by_pid[pid].image, sz)
        if person_img is not None:
            grid2.paste(Image.fromarray(person_img), (0, r * hh))
        ref = load(Path(f"{stem(args.sets[0], pid, gid)}.ref.png"), sz)
        if ref is not None:
            grid2.paste(Image.fromarray(ref), (TH, r * hh))
        for i, s in enumerate(args.sets):
            a = load(Path(f"{stem(s, pid, gid)}.png"), sz)
            if a is not None:
                grid2.paste(Image.fromarray(a), ((2 + i) * TH, r * hh))
    heads2 = ["person", "garment reference"] + [s.replace("ckpt_control_on", "output")
                                                for s in args.sets]
    f2 = Image.new("RGB", (grid2.width, grid2.height + 26), "white")
    f2.paste(label_strip(grid2.width, heads2, TH), (0, 0))
    f2.paste(grid2, (0, 26))
    p2 = fig_dir / f"fig_refs__{'_vs_'.join(args.sets)}.png"
    f2.save(p2)
    print(f"[2] {p2}  ({len(rows)} combos)")
    print("\nNOTE: the reference panel is taken from the first set; if the sets are in different "
          "regimes (textured vs grey), state that in the figure caption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())