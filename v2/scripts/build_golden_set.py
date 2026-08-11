#!/usr/bin/env python3
"""Golden set construction: a 20 people × 5 garments × 4 angles manifest + image copies.

Usage (on Colab, with the data present):
  python v2/scripts/build_golden_set.py \
      --vitonhd-test data/zalando-hd-resized/test/image \
      --garments data/garments_3d \
      [--person-list v2/configs/golden_persons.txt] \
      [--garment-ids id1,id2,id3,id4,id5]

- If --person-list is not given, 20 evenly spaced images are picked from the test set and a
  WARNING is printed: the list must be edited BY HAND for side/¾ poses and 2-3 non-VITON photos.
- Garments: since the appearance ref is ALWAYS converted to flat grey in the builder (the
  PERMANENT textureless rule), textured/untextured is not distinguished; any valid mesh works.
- The images are copied under v2/data/golden/persons/ (gitignored); manifest.json is
  committed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import yaml  # noqa: E402

from meshvton2.eval.golden_set import GoldenGarment, GoldenManifest, GoldenPerson  # noqa: E402

ANGLES = [0, 90, 180, 270]
NUM_PERSONS = 20
NUM_GARMENTS = 5


def find_texture(garment_dir: Path) -> Path | None:
    """Same rule as v1 garment_draper._find_sibling_texture: the first PNG in the folder."""
    pngs = sorted(garment_dir.glob("*.png")) + sorted(garment_dir.glob("*.jpg"))
    return pngs[0] if pngs else None


def pick_garments(garments_root: Path, ids: list[str] | None) -> list[GoldenGarment]:
    """Finds the garment folders RECURSIVELY (CLOTH3D layout: category/garment/model.obj).
    Automatic selection happens ONLY under the upper_body folder (the drape logic is tuned
    for the upper body; other splits may contain unchecked Trousers/Skirt/Dress types);
    a texture is NOT required (the appearance ref is converted to grey in the builder
    either way)."""
    out = []
    if ids:
        dirs = [garments_root / gid for gid in ids]
    else:
        dirs = sorted(
            d for d in {p.parent for p in garments_root.rglob("*.obj")}
            if "upper_body" in d.relative_to(garments_root).parts
        )
    for d in dirs:
        objs = sorted(d.glob("*.obj"))
        if not objs:
            print(f"WARNING: no .obj inside {d}, skipping", file=sys.stderr)
            continue
        tex = find_texture(d)
        rel = d.relative_to(garments_root)
        out.append(
            GoldenGarment(
                id=str(rel).replace("/", "__"),  # used in file names, cannot contain /
                mesh=str(objs[0].relative_to(garments_root)),
                texture=str(tex.relative_to(garments_root)) if tex else None,
                category=rel.parts[0] if len(rel.parts) > 1 else "top",
            )
        )
        if len(out) == NUM_GARMENTS and not ids:
            break
    return out


def pick_persons(test_dir: Path, list_file: Path | None) -> list[Path]:
    if not test_dir.exists():
        raise SystemExit(
            f"ERROR: no person image directory: {test_dir}\n"
            "The Colab disk resets every session — run the DATA cell (unzip) first."
        )
    if list_file and list_file.exists():
        return [test_dir / line.strip() for line in list_file.read_text().splitlines() if line.strip()]
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in exts)
    if not images:  # nested zip layouts: if the root is empty, search recursively
        images = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in exts)
        if images:
            print(f"NOTE: images found in a subfolder: {images[0].parent}", file=sys.stderr)
    step = max(1, len(images) // NUM_PERSONS)
    picked = images[::step][:NUM_PERSONS]
    print(
        "WARNING: --person-list was not given; an evenly spaced automatic selection was made.\n"
        "The golden set needs MANUAL curation for pose diversity (side/¾) and 2-3 non-VITON photos:\n"
        f"  write the selection to v2/configs/golden_persons.txt and edit it.",
        file=sys.stderr,
    )
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vitonhd-test", type=Path, required=True)
    ap.add_argument("--garments", type=Path, required=True)
    ap.add_argument("--person-list", type=Path, default=REPO / "v2/configs/golden_persons.txt")
    ap.add_argument("--garment-ids", default=None, help="5 comma-separated garment folder names")
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    golden_root = REPO / base["paths"]["golden_root"]
    (golden_root / "persons").mkdir(parents=True, exist_ok=True)

    person_paths = pick_persons(args.vitonhd_test, args.person_list)
    persons = []
    for p in person_paths:
        if not p.exists():
            print(f"WARNING: {p} does not exist, skipping", file=sys.stderr)
            continue
        dst = golden_root / "persons" / p.name
        shutil.copy(p, dst)
        persons.append(GoldenPerson(id=p.stem, image=f"persons/{p.name}", source="vitonhd_test"))

    garment_ids = args.garment_ids.split(",") if args.garment_ids else None
    garments = pick_garments(args.garments, garment_ids)

    if len(persons) < NUM_PERSONS or len(garments) < NUM_GARMENTS:
        print(
            f"WARNING: target {NUM_PERSONS} people × {NUM_GARMENTS} garments; "
            f"found {len(persons)} × {len(garments)}",
            file=sys.stderr,
        )
    if not persons or not garments:
        print("ERROR: the golden set is empty — check the paths", file=sys.stderr)
        return 1

    manifest = GoldenManifest(root=golden_root, persons=persons, garments=garments, angles=ANGLES)
    manifest.save(golden_root / "manifest.json")
    n = len(manifest.combos) * len(ANGLES)
    print(f"OK: {len(persons)} people × {len(garments)} garments × {len(ANGLES)} angles = {n} evaluation items")
    print(f"Manifest: {golden_root / 'manifest.json'} (committed; the images are gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
