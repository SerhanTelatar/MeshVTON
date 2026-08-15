#!/usr/bin/env python3
"""Builds the CONDITIONING figure: everything the pipeline hands the diffusion model, side by side.

The paper describes the conditioning across three subsections of prose; this shows it in one row.
Panels, left to right: input photograph, agnostic person, inpainting mask, rendered normal map,
rendered depth+silhouette, appearance reference. The shape is wide and short, which is what a
two-column \\begin{figure*} at the top of a page wants.

NO diffusion: parser + HMR2 + render only, a few seconds per person once the models are loaded.

Usage (Colab):
  python v2/scripts/fig_conditioning.py --idm-repo /content/IDM-VTON
  [--persons 00000_00 01455_00] [--garment upper_body__00111_Tshirt] [--thumb 220]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from meshvton2.conditioning.body import build_hmr2_backend  # noqa: E402
from meshvton2.conditioning.builder import (  # noqa: E402
    PHOTO_GARMENT_SCALE, PHOTO_HANG_PAD, PhotoView, assert_real_impl, build_conditioning)
from meshvton2.conditioning.garment import load_garment_asset  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.utils.image_utils import tensor_to_pil  # noqa: E402

# Column headers, in the order the panels are drawn. Kept short: they are printed at figure scale.
# Column key -> header text, in the order they are produced. --panels selects a subset.
PANELS = {
    "photo": "input photograph",
    "agnostic": "agnostic",
    "mask": "inpainting mask",
    "normal": "normal map",
    "depth": "depth + silhouette",
    "ref": "appearance reference",
}

# Persons whose camera fit passed the gate in calibrate_hang.py — on these the projected body
# demonstrably lands on the photograph, so the geometry panels illustrate the method rather than
# a regression failure.
CAMERA_VERIFIED = ("00000_00", "02935_00", "01455_00", "00737_00", "02199_00")

ATR_GARMENT = (4, 7)  # upper-body labels in the parser, as used by calibrate_hang.py


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--persons", nargs="+", default=None,
                    help="default: the first two persons whose camera fit was verified")
    ap.add_argument("--garments", nargs="+", default=None,
                    help="one garment per row; a single value is reused for every row. "
                         "Default: a different qualifying garment per row, so the appearance "
                         "column carries information instead of repeating")
    ap.add_argument("--rank", type=int, default=0, metavar="N",
                    help="score every person x garment combination by mesh-parser silhouette IoU "
                         "(the measure the calibration uses) and draw the best N as rows. "
                         "Selection is then by measurement rather than by eye")
    ap.add_argument("--panels", nargs="+", default=None, metavar="NAME",
                    help=f"which columns to draw, in order. Available: {', '.join(PANELS)}. "
                         "Six panels need a two-column figure*; for a single column drop to four")
    ap.add_argument("--thumb", type=int, default=220)
    ap.add_argument("--use-texture", action="store_true",
                    help="render the appearance reference WITH its texture. Match whatever the "
                         "evaluation run used, or the figure will not depict the reported system")
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    garments_root = REPO / base["paths"]["garments_root"]
    fig_dir = REPO / base["paths"]["eval_results"] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_pid = {p.id: p for p in manifest.persons}
    by_gid = {g.id: g for g in manifest.garments}
    # The default persons are the ones whose camera fit was verified (calibrate_hang.py): on those
    # the mesh demonstrably projects onto the body, which is what this figure is meant to show.
    verified = [p for p in CAMERA_VERIFIED if p in by_pid]
    pids = args.persons or (verified or [p.id for p in manifest.persons])[:2]

    def load(g):
        p = garments_root / g.mesh if g.mesh else None
        if not (p and p.exists()):
            return None
        return load_garment_asset(
            p, texture_path=garments_root / g.texture if g.texture else None,
            garment_id=g.id, allow_untextured=True)

    # render_appearance_ref needs BOTH a texture and UV coordinates (render.py::render_appearance_ref).
    # A manifest entry can name a texture whose mesh still carries no UVs, so decide on the LOADED
    # asset rather than on the manifest, and walk the candidates until one actually qualifies.
    def usable(a) -> bool:
        return a is not None and a.texture is not None and a.uv is not None

    def pick(order) -> list:
        """Loads garments in `order`, keeping those this run can actually render."""
        out = []
        for g in order:
            a = load(g)
            if a is None or (args.use_texture and not usable(a)):
                continue
            out.append((g.id, a))
        return out

    if args.garments:
        chosen = pick([by_gid[g] for g in args.garments])
        if len(chosen) < len(args.garments):
            bad = set(args.garments) - {gid for gid, _ in chosen}
            raise SystemExit(f"ERROR: cannot render with {sorted(bad)} under the current flags")
        if len(chosen) == 1:
            chosen *= len(pids)
    else:
        chosen = pick(manifest.garments)

    if not chosen:
        if args.use_texture:
            raise SystemExit(
                "ERROR: --use-texture was given but no garment in the manifest loads with both a "
                f"texture and UV coordinates (searched {len(manifest.garments)} under "
                f"{garments_root}). Drop --use-texture to render the grey shaded reference, or "
                "restore a manifest that records textures.")
        print(f"NOTE: no garment mesh found under {garments_root} — falling back to body-only "
              f"geometry; the silhouette panel will show the body only.", file=sys.stderr)
        chosen = [("(body only)", None)]

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()

    # Person preprocessing and the body regression depend only on the photograph, so they are
    # done once per person and reused across every garment.
    people: dict[str, tuple] = {}

    def prepare(pid: str):
        if pid not in people:
            pp = prep.process(manifest.root / by_pid[pid].image, size=size)
            worn = np.isin(cv2.resize(np.asarray(pp.parse), (size[1], size[0]),
                                      interpolation=cv2.INTER_NEAREST), ATR_GARMENT)
            people[pid] = (pp, hmr2(pp.image, bbox=person_square_bbox(pp)), worn)
        return people[pid]

    def build(pid: str, asset):
        pp, params, _ = prepare(pid)
        kw = dict(size=size, person_prep=pp, use_texture=args.use_texture)
        if asset is not None:
            kw.update(hang_pad=PHOTO_HANG_PAD, garment_scale=PHOTO_GARMENT_SCALE)
        return build_conditioning(pp.image, params, asset, PhotoView(), **kw)

    if args.rank:
        # Score every combination the way the calibration does: IoU between the rendered garment
        # silhouette and the garment region the parser finds on the photograph. That reference is
        # independent of the diffusion model, so the ranking measures conditioning quality rather
        # than output quality, and the rows are chosen by measurement rather than by eye.
        cand_pids = args.persons or [p.id for p in manifest.persons]
        scored = []
        for pid in cand_pids:
            if pid not in by_pid:
                continue
            _, _, worn = prepare(pid)
            for gid, asset in chosen:
                sil = build(pid, asset).control_depth_sil[2].numpy() > 0
                union = (sil | worn).sum()
                scored.append((float((sil & worn).sum() / union) if union else 0.0, pid, gid, asset))
        scored.sort(reverse=True, key=lambda r: r[0])
        print("\ntop combinations by mesh-parser IoU:")
        for iou, pid, gid, _ in scored[: args.rank]:
            print(f"  {iou:.3f}  {pid} x {gid}")
        plan = [(pid, (gid, asset)) for _, pid, gid, asset in scored[: args.rank]]
    else:
        # One garment per row, cycling if fewer garments than persons.
        plan = list(zip(pids, [chosen[i % len(chosen)] for i in range(len(pids))]))
        for pid, (gid, _) in plan:
            print(f"row: {pid} x {gid}")

    keys = args.panels or list(PANELS)
    bad = [k for k in keys if k not in PANELS]
    if bad:
        raise SystemExit(f"ERROR: unknown panel {bad}; available: {', '.join(PANELS)}")

    TH = args.thumb
    hh = round(TH * size[0] / size[1])
    HEAD = 24
    rows = []

    for pid, (gid, asset) in plan:
        if pid not in by_pid:
            print(f"SKIP {pid}: not in the manifest", file=sys.stderr)
            continue
        pp, _, _ = prepare(pid)
        b = build(pid, asset)

        avail = {
            "photo": Image.fromarray(pp.image),
            "agnostic": tensor_to_pil(b.agnostic_rgb),
            "mask": Image.fromarray((b.inpaint_mask.numpy()[0] * 255).astype("uint8")),
            "normal": tensor_to_pil(b.control_normal),
            "depth": tensor_to_pil(b.control_depth_sil),
            "ref": tensor_to_pil(b.appearance_ref),
        }
        rows.append([avail[k].convert("RGB").resize((TH, hh), Image.LANCZOS) for k in keys])
        print(f"OK {pid} x {gid}")

    if not rows:
        raise SystemExit("ERROR: no person could be prepared")

    im = Image.new("RGB", (TH * len(keys), HEAD + hh * len(rows)), "white")
    d = ImageDraw.Draw(im)
    for i, k in enumerate(keys):
        d.text((i * TH + 5, 7), PANELS[k], fill="black")
    for r, panels in enumerate(rows):
        for c, p in enumerate(panels):
            im.paste(p, (c * TH, HEAD + r * hh))

    out = fig_dir / "fig_conditioning.png"
    im.save(out)
    print(f"\n{out}  ({im.width}x{im.height})")
    print("Place it as a two-column \\begin{figure*}[t] near the top of the paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
