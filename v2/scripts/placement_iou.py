#!/usr/bin/env python3
"""YERLEŞİM metriği — üretilen giysi GERÇEKTEN gövdeye oturmuş mu?

Neden gerekli (2026-08-10 bulgusu): harness'taki `geo_iou` çıktıyı, koşullama
olarak VERDİĞİMİZ silüetle kıyaslar. Yani ölçtüğü şey "sonuç doğru mu" değil,
"model verdiğim silüeti takip etti mi". Silüeti yanlış yere (yüzün üstüne)
koyarsak model oraya çizer ve geo_iou yine yüksek çıkar — metrik hedefin
kendisinin bozuk olduğunu GÖREMEZ. Askı düzeltmesi (hang_pad -0.12) mesh↔parser
hizasını 0.295→0.480 yükselttiği hâlde geo_iou'nun düşmesi tam olarak budur.

Bu script bağımsız bir referans kullanır: kişinin ÜZERİNDEKİ gerçek giysinin
parser maskesi. Üretilen giysi bölgesi oraya ne kadar oturuyor?
  placement_iou = IoU(predsil, parser_worn_mask)

Kesim farkları yüzünden 1.0'a çıkamaz (farklı giysi mesh'i giydiriyoruz), ama
koşular ARASINDA kıyaslanabilir — hangi hang_pad daha gerçekçi yerleştiriyor.

GPU: yalnız parser (difüzyon yok).

Kullanım:
  python v2/scripts/placement_iou.py --idm-repo /content/IDM-VTON \\
      [--sets ckpt_control_on ckpt_control_on_hp-012]
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
from PIL import Image  # noqa: E402

from meshvton2.conditioning.person import PersonPreprocessor  # noqa: E402
from meshvton2.eval.golden_set import load_manifest  # noqa: E402
from meshvton2.eval.harness import _load_mask  # noqa: E402

ATR_GARMENT = (4, 7)
DEFAULT_SETS = ("phase1_fill_spatial", "ckpt_control_on", "ckpt_control_off",
                "ckpt_control_on_hp-012", "ckpt_control_off_hp-012")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, required=True)
    ap.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS))
    ap.add_argument("--angles", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load((REPO / "v2/configs/eval.yaml").read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])
    manifest = load_manifest(REPO / cfg["golden_manifest"])
    out_root = REPO / base["paths"]["eval_results"]
    H, W = size

    prep = PersonPreprocessor(args.idm_repo)
    by_pid = {p.id: p for p in manifest.persons}

    # Kişi başına GERÇEK giysi maskesi bir kez (tüm setler aynı referansı kullansın)
    worn: dict[str, np.ndarray] = {}

    def worn_mask(pid: str) -> np.ndarray | None:
        if pid not in worn:
            try:
                pp = prep.process(manifest.root / by_pid[pid].image, size=size)
                worn[pid] = np.isin(cv2.resize(np.asarray(pp.parse), (W, H),
                                               interpolation=cv2.INTER_NEAREST), ATR_GARMENT)
            except Exception as e:
                print(f"  {pid}: parser başarısız — {e}", file=sys.stderr)
                worn[pid] = None
        return worn[pid]

    print("=== YERLEŞİM IoU (üretilen giysi vs kişinin gerçek giysisi) ===")
    print("    geo_iou'dan FARKI: hedef bizim koyduğumuz silüet değil, bağımsız parser.\n")
    rows = []
    for name in args.sets:
        pred_dir = out_root / name / "preds"
        if not pred_dir.exists():
            print(f"{name:28s} klasör yok — atlandı")
            continue
        vals = []
        for combo in manifest.combos:
            for angle in args.angles:
                stem = (pred_dir / cfg["pred_pattern"].format(
                    person_id=combo.person_id, garment_id=combo.garment_id,
                    angle=angle)).with_suffix("")
                ps = Path(f"{stem}.predsil.png")
                if not ps.exists():
                    continue
                w = worn_mask(combo.person_id)
                if w is None:
                    continue
                p = _load_mask(ps, size=w.shape)
                union = (p | w).sum()
                if union:
                    vals.append(float((p & w).sum() / union))
        if not vals:
            print(f"{name:28s} .predsil yok — atlandı")
            continue
        rows.append((name, float(np.mean(vals)), len(vals)))
        print(f"{name:28s} yerleşim IoU = {np.mean(vals):.4f}  (n={len(vals)})")

    if len(rows) >= 2:
        best = max(rows, key=lambda r: r[1])
        print(f"\nEN İYİ YERLEŞİM: {best[0]} → {best[1]:.4f}")
        print("Bu metrik hangi koşullamanın GERÇEKÇİ yerleştirdiğini söyler;")
        print("geo_iou ise yalnız modelin koşullamaya sadakatini ölçer. İkisi çelişirse")
        print("görsel doğruluk için BU metriğe bakın.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
