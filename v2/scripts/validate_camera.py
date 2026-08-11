#!/usr/bin/env python3
"""Phase 2 exit gate — camera validation (Colab, GPU).

For the golden set people: HMR2 → SMPL-X → render the body silhouette with the pred_cam
camera → IoU against the parser's person mask. Gate: mean IoU >= 0.70.
It writes an overlay PNG per person (green=parser, red=SMPL-X reprojection,
yellow=intersection) — misalignment is immediately visible.

Usage:
  python v2/scripts/validate_camera.py --idm-repo /content/IDM-VTON [--limit 5]

Requirements: 4D-Humans (pip install git+https://github.com/shubham-goel/4D-Humans.git)
+ smplx + SMPLX_MODEL_DIR + IDM-VTON preprocess (parsing onnx).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from meshvton2.conditioning.body import build_hmr2_backend, get_body_model  # noqa: E402
from meshvton2.conditioning.builder import assert_real_impl  # noqa: E402
from meshvton2.conditioning.camera import photo_camera  # noqa: E402
from meshvton2.conditioning.person import PersonPreprocessor, person_square_bbox  # noqa: E402
from meshvton2.conditioning.render import render_body_mask  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402

IOU_GATE = 0.70


def overlay(person: np.ndarray, parse_mask: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    out = (person * 0.45).astype(np.uint8)
    out[parse_mask & ~body_mask] = (40, 200, 40)    # parser only: green
    out[body_mask & ~parse_mask] = (220, 50, 50)    # SMPL-X only: red
    out[parse_mask & body_mask] = (230, 220, 60)    # intersection: yellow
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    assert_real_impl()
    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(args.manifest or REPO / cfg["golden_manifest"])
    out_dir = REPO / base["paths"]["eval_results"] / "camera_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = PersonPreprocessor(args.idm_repo)
    hmr2 = build_hmr2_backend()
    body_model = get_body_model()

    persons = manifest.persons[: args.limit] if args.limit else manifest.persons
    ious = []
    for i, person in enumerate(persons):
        pp = prep.process(manifest.root / person.image, size=size)
        params = hmr2(pp.image, bbox=person_square_bbox(pp))  # person-centred square (alignment)
        body = body_model(params)
        cam = photo_camera(params, size)
        body_mask = render_body_mask(body["verts"], body["faces"], cam, size=size)
        # the parser's person mask: every label other than background(0)
        parse_full = np.asarray(
            Image.fromarray(pp.parse).resize((size[1], size[0]), Image.NEAREST)
        )
        parse_mask = parse_full > 0
        union = (parse_mask | body_mask).sum()
        iou = float((parse_mask & body_mask).sum() / union) if union else 0.0
        ious.append(iou)
        Image.fromarray(overlay(pp.image, parse_mask, body_mask)).save(
            out_dir / f"{person.id}_iou{iou:.2f}.png"
        )
        print(f"[{i+1}/{len(persons)}] {person.id}: IoU={iou:.3f}")

    mean_iou = float(np.mean(ious))
    verdict = "PASSED" if mean_iou >= IOU_GATE else "FAILED"
    print(f"\nMean IoU = {mean_iou:.3f} (gate {IOU_GATE}) → {verdict}")
    print(f"Overlays: {out_dir} (green=parser, red=SMPL-X, yellow=intersection)")
    return 0 if mean_iou >= IOU_GATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
