#!/usr/bin/env python3
"""Golden set inşası: 20 kişi × 5 giysi × 4 açı manifest'i + görüntü kopyaları.

Kullanım (Colab'da, veri mevcutken):
  python v2/scripts/build_golden_set.py \
      --vitonhd-test data/zalando-hd-resized/test/image \
      --garments data/garments_3d \
      [--person-list v2/configs/golden_persons.txt] \
      [--garment-ids id1,id2,id3,id4,id5]

- --person-list verilmezse test setinden eşit aralıklı 20 görüntü seçilir ve
  UYARI basılır: yan/¾ pozlar ve 2-3 VITON-dışı foto için liste ELLE düzenlenmeli.
- Giysiler: sibling texture PNG'si olan klasörler tercih edilir (v1 Probe D dersi:
  texturesiz mesh golden set'e giremez).
- Görüntüler v2/data/golden/persons/ altına kopyalanır (gitignored); manifest.json
  commit'lenir.
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
    """v1 garment_draper._find_sibling_texture ile aynı kural: klasördeki ilk PNG."""
    pngs = sorted(garment_dir.glob("*.png")) + sorted(garment_dir.glob("*.jpg"))
    return pngs[0] if pngs else None


def pick_garments(garments_root: Path, ids: list[str] | None) -> list[GoldenGarment]:
    out = []
    if ids:
        dirs = [garments_root / gid for gid in ids]
    else:
        dirs = sorted(d for d in garments_root.iterdir() if d.is_dir() and list(d.glob("*.obj")))
    for d in dirs:
        objs = sorted(d.glob("*.obj"))
        if not objs:
            print(f"UYARI: {d} içinde .obj yok, atlanıyor", file=sys.stderr)
            continue
        tex = find_texture(d)
        if tex is None and not ids:
            continue  # otomatik seçimde texturesiz giysi golden set'e giremez
        out.append(
            GoldenGarment(
                id=d.name,
                mesh=str(objs[0].relative_to(garments_root)),
                texture=str(tex.relative_to(garments_root)) if tex else None,
            )
        )
        if len(out) == NUM_GARMENTS and not ids:
            break
    return out


def pick_persons(test_dir: Path, list_file: Path | None) -> list[Path]:
    if list_file and list_file.exists():
        return [test_dir / line.strip() for line in list_file.read_text().splitlines() if line.strip()]
    images = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    step = max(1, len(images) // NUM_PERSONS)
    picked = images[::step][:NUM_PERSONS]
    print(
        "UYARI: --person-list verilmedi; eşit aralıklı otomatik seçim yapıldı.\n"
        "Golden set poz çeşitliliği (yan/¾) ve VITON-dışı 2-3 foto için ELLE küratörlük ister:\n"
        f"  seçimi v2/configs/golden_persons.txt'e yazıp düzenleyin.",
        file=sys.stderr,
    )
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vitonhd-test", type=Path, required=True)
    ap.add_argument("--garments", type=Path, required=True)
    ap.add_argument("--person-list", type=Path, default=REPO / "v2/configs/golden_persons.txt")
    ap.add_argument("--garment-ids", default=None, help="Virgülle ayrık 5 giysi klasör adı")
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    golden_root = REPO / base["paths"]["golden_root"]
    (golden_root / "persons").mkdir(parents=True, exist_ok=True)

    person_paths = pick_persons(args.vitonhd_test, args.person_list)
    persons = []
    for p in person_paths:
        if not p.exists():
            print(f"UYARI: {p} yok, atlanıyor", file=sys.stderr)
            continue
        dst = golden_root / "persons" / p.name
        shutil.copy(p, dst)
        persons.append(GoldenPerson(id=p.stem, image=f"persons/{p.name}", source="vitonhd_test"))

    garment_ids = args.garment_ids.split(",") if args.garment_ids else None
    garments = pick_garments(args.garments, garment_ids)

    if len(persons) < NUM_PERSONS or len(garments) < NUM_GARMENTS:
        print(
            f"UYARI: hedef {NUM_PERSONS} kişi × {NUM_GARMENTS} giysi; "
            f"bulunan {len(persons)} × {len(garments)}",
            file=sys.stderr,
        )
    if not persons or not garments:
        print("HATA: golden set boş — yolları kontrol edin", file=sys.stderr)
        return 1

    manifest = GoldenManifest(root=golden_root, persons=persons, garments=garments, angles=ANGLES)
    manifest.save(golden_root / "manifest.json")
    n = len(manifest.combos) * len(ANGLES)
    print(f"OK: {len(persons)} kişi × {len(garments)} giysi × {len(ANGLES)} açı = {n} değerlendirme öğesi")
    print(f"Manifest: {golden_root / 'manifest.json'} (commit'lenir; görüntüler gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
