#!/usr/bin/env python3
"""Mesh özgüllük MATRİSİ figürü — tezin ana görsel kanıtı.

Bir kişi için N×N ısı haritası: satır = i. giysiyle ÜRETİLEN çıktının silüeti,
sütun = j. giysinin HEDEF silüeti, hücre = IoU. Köşegen (doğru eşleşme) parlaksa
model verilen mesh'e özeldir. `mesh_specificity.py` bu sayıyı hesaplıyordu;
bu script onu görselleştirir.

Difüzyon YOK — mevcut .predsil.png / .sil.png dosyalarını okur.

Kullanım:
  python v2/scripts/specificity_matrix.py --sets ckpt_control_on_july ckpt_control_off_july
  [--person 00000_00] [--cell 90]
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

from meshvton2.eval import metrics as M  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.eval.harness import _load_mask  # noqa: E402


def heat(v: float, lo: float, hi: float) -> tuple[int, int, int]:
    """Düşük = koyu lacivert, yüksek = sarı (algısal olarak monoton, gri baskıda da okunur)."""
    t = 0.0 if hi <= lo else float(np.clip((v - lo) / (hi - lo), 0, 1))
    return (int(30 + 225 * t ** 0.8), int(30 + 195 * t ** 1.1), int(90 - 60 * t))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--person", default=None, help="varsayılan: ilk kişi")
    ap.add_argument("--angle", type=int, default=0)
    ap.add_argument("--cell", type=int, default=90, help="hücre kenarı [px]")
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    out_root = REPO / base["paths"]["eval_results"]
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pid = args.person or manifest.combos[0].person_id
    gids = list(dict.fromkeys(c.garment_id for c in manifest.combos if c.person_id == pid))
    n = len(gids)
    C, PAD = args.cell, 150

    panels = []
    for s in args.sets:
        pred_dir = out_root / s / "preds"
        stem = lambda gid: (pred_dir / cfg["pred_pattern"].format(
            person_id=pid, garment_id=gid, angle=args.angle)).with_suffix("")
        mat = np.full((n, n), np.nan)
        for i, gi in enumerate(gids):
            ps = Path(f"{stem(gi)}.predsil.png")
            if not ps.exists():
                continue
            a = _load_mask(ps)
            for j, gj in enumerate(gids):
                sl = Path(f"{stem(gj)}.sil.png")
                if not sl.exists():
                    continue
                v = M.silhouette_iou(a, _load_mask(sl, size=a.shape))
                if v is not None:
                    mat[i, j] = v
        if np.all(np.isnan(mat)):
            print(f"{s}: .predsil/.sil yok — atlandı", file=sys.stderr)
            continue
        diag = np.nanmean(np.diag(mat))
        off = np.nanmean(mat[~np.eye(n, dtype=bool)])
        panels.append((s, mat, diag, off))
        print(f"{s}: köşegen={diag:.4f} köşegen-dışı={off:.4f} ayırt edicilik={diag-off:+.4f}")

    if not panels:
        raise SystemExit("HATA: hiç matris kurulamadı")

    lo = min(np.nanmin(m) for _, m, _, _ in panels)
    hi = max(np.nanmax(m) for _, m, _, _ in panels)
    W = PAD + n * C + 20
    im = Image.new("RGB", (W * len(panels), PAD + n * C + 60), "white")
    d = ImageDraw.Draw(im)
    short = [g.split("__")[-1][:11] for g in gids]
    for k, (s, mat, diag, off) in enumerate(panels):
        x0 = k * W + PAD
        d.text((k * W + 8, 14), s.replace("ckpt_", ""), fill="black")
        d.text((k * W + 8, 30), f"ayirt edicilik {diag - off:+.4f}", fill="black")
        for j, t in enumerate(short):  # sütun başlıkları (HEDEF silüet)
            d.text((x0 + j * C + 4, PAD - 22), t[:9], fill="black")
        for i, t in enumerate(short):  # satır başlıkları (ÜRETİLEN çıktı)
            d.text((k * W + 6, PAD + i * C + C // 2 - 6), t, fill="black")
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                box = [x0 + j * C, PAD + i * C, x0 + (j + 1) * C - 2, PAD + (i + 1) * C - 2]
                if np.isnan(v):
                    d.rectangle(box, fill=(230, 230, 230)); continue
                d.rectangle(box, fill=heat(v, lo, hi),
                            outline=(255, 255, 255) if i == j else None,
                            width=3 if i == j else 0)
                d.text((box[0] + C // 2 - 16, box[1] + C // 2 - 6), f"{v:.2f}",
                       fill="white" if v < (lo + hi) / 2 else "black")
        d.text((x0, PAD + n * C + 8), "sütun = HEDEF mesh silüeti", fill="black")
        d.text((x0, PAD + n * C + 26), "satır = ÜRETİLEN çıktının silüeti", fill="black")

    out = fig_dir / f"fig_specificity_matrix__{pid}.png"
    im.save(out)
    print(f"\n{out}\nKöşegen (beyaz çerçeveli) parlaksa model verilen mesh'e ÖZELDİR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())