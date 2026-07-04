#!/usr/bin/env python3
"""Colab TEK-KOŞU pipeline'ı — bakıcılık istemez; her aşama idempotent/resume'ludur.

Aşamalar (sırayla; biten aşama tekrar koşulmaz):
  1. check     : GPU/VRAM/ortam kontrolü (eğitim için >=20GB ister; yoksa veri-only mod)
  2. camera    : Faz 2 kapısı — reprojection IoU >= 0.70 (KALIRSA DURUR)
  3. smoke     : sentetik duman testi (5 örnek) + QA kontak-sheet'i Drive'a
  4. synth     : tam sentetik üretim (--num, varsayılan 2000; 4 görüş/örnek)
  5. vitonhd   : gerçek veri ön-işleme (--vitonhd-limit)
  6. baseline  : Faz 1 zero-shot taban çizgisi (fill_spatial, tam golden set)
  7. train     : Aşama-1 eğitim (resume'lu; Colab koparsa tekrar Run all yeter)
  8. eval      : golden set değerlendirme + kontrol-ablation kapısı

Kullanım (notebook tek hücre):
  python v2/scripts/colab_pipeline.py --idm-repo /content/IDM-VTON \\
      [--until train] [--num 2000] [--vitonhd-limit 2000] [--skip baseline]

Her aşama sonrası kritik çıktılar Drive'a kopyalanır (mount edilmişse):
  /content/drive/MyDrive/MeshVTON/v2_outputs/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRIVE_OUT = Path("/content/drive/MyDrive/MeshVTON/v2_outputs")
STAGES = ("check", "camera", "smoke", "synth", "vitonhd", "baseline", "train", "eval")


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n== {msg}\n{'=' * 70}", flush=True)


def run(cmd: list[str], gate: bool = False) -> int:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.call([sys.executable, *cmd[1:]] if cmd[0] == "python" else cmd)
    if rc != 0 and gate:
        raise SystemExit(f"KAPI KALDI (rc={rc}): {' '.join(map(str, cmd))} — pipeline durdu.")
    return rc


def sync_drive(*paths: Path) -> None:
    if not DRIVE_OUT.parent.parent.exists():  # drive mount edilmemiş
        return
    for p in paths:
        if not p.exists():
            continue
        dst = DRIVE_OUT / p.name
        if p.is_dir():
            shutil.copytree(p, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    print(f"→ Drive'a kopyalandı: {DRIVE_OUT}", flush=True)


def gpu_info() -> tuple[str, float]:
    import torch

    if not torch.cuda.is_available():
        return "YOK", 0.0
    p = torch.cuda.get_device_properties(0)
    return p.name, p.total_memory / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, default=Path("/content/IDM-VTON"))
    ap.add_argument("--garments", type=Path, default=REPO / "data/garments_3d")
    ap.add_argument("--images", type=Path, default=REPO / "data/raw/images")
    ap.add_argument("--cloth", type=Path, default=REPO / "data/raw/cloth")
    ap.add_argument("--poses", type=Path, default=REPO / "data/smplx_params")
    ap.add_argument("--num", type=int, default=2000, help="sentetik örnek sayısı")
    ap.add_argument("--vitonhd-limit", type=int, default=2000)
    ap.add_argument("--train-steps", type=int, default=20000)
    ap.add_argument("--until", choices=STAGES, default="eval")
    ap.add_argument("--skip", action="append", default=[], choices=STAGES)
    args = ap.parse_args()

    until = STAGES.index(args.until)
    eval_dir = REPO / "v2/eval_results"
    synth_dir = REPO / "v2/data/synth"
    can_train = True

    def active(stage: str) -> bool:
        return STAGES.index(stage) <= until and stage not in args.skip

    # ---- 1. check ----
    banner("1/8 ortam kontrolü")
    name, vram = gpu_info()
    print(f"GPU: {name} ({vram:.0f} GB)")
    if vram < 20:
        can_train = False
        print("UYARI: <20GB VRAM — FLUX aşamaları (baseline/train) ATLANACAK; "
              "veri üretimi + doğrulama yine koşar. Eğitim için L4/A100 runtime seçin.")
    sil = shutil.which("python") or sys.executable
    print(f"python: {sil}")

    # ---- 2. camera gate ----
    if active("camera"):
        banner("2/8 kamera doğrulama kapısı (IoU >= 0.70)")
        if not (REPO / "v2/data/golden/manifest.json").exists():
            run(["python", "v2/scripts/build_golden_set.py",
                 "--vitonhd-test", str(args.images), "--garments", str(args.garments)], gate=True)
        run(["python", "v2/scripts/validate_camera.py", "--idm-repo", str(args.idm_repo), "--limit", "5"],
            gate=True)
        sync_drive(eval_dir / "camera_validation")

    # ---- 3. smoke + QA ----
    if active("smoke"):
        banner("3/8 sentetik duman testi + QA sheet")
        if not (synth_dir / "pairs.csv").exists():
            run(["python", "v2/scripts/generate_synthetic.py", "--garments", str(args.garments),
                 "--poses", str(args.poses), "--num", "5", "--limit-garments", "3"], gate=True)
        qa = REPO / "v2/eval_results/synth_qa"
        qa.mkdir(parents=True, exist_ok=True)
        first = sorted(synth_dir.glob("s*_*/"))
        if first:
            shutil.copytree(first[0], qa / first[0].name, dirs_exist_ok=True)
        sync_drive(qa)
        print("QA görselleri Drive'da: v2_outputs/synth_qa — view_180/gt.png'yi GÖZLE kontrol edin.")

    # ---- 4. full synth ----
    if active("synth"):
        banner(f"4/8 tam sentetik üretim ({args.num} örnek × 4 görüş)")
        existing = sum(1 for _ in synth_dir.glob("s*_*/")) if synth_dir.exists() else 0
        if existing >= args.num:
            print(f"zaten {existing} örnek var, atlanıyor")
        else:
            run(["python", "v2/scripts/generate_synthetic.py", "--garments", str(args.garments),
                 "--poses", str(args.poses), "--num", str(args.num - existing),
                 "--seed", str(existing)], gate=True)
        sync_drive(synth_dir / "pairs.csv")

    # ---- 5. vitonhd ----
    if active("vitonhd"):
        banner("5/8 VITON-HD ön-işleme")
        if not args.cloth.exists():
            print(f"UYARI: {args.cloth} yok (VITON-HD cloth/ ürün fotoğrafları) — "
                  "gerçek veri ATLANIYOR; eğitim sentetik-only olur (domain-gap riski).")
        else:
            run(["python", "v2/scripts/preprocess_vitonhd.py", "--images", str(args.images),
                 "--cloth", str(args.cloth), "--idm-repo", str(args.idm_repo),
                 "--limit", str(args.vitonhd_limit)], gate=False)

    # ---- 6. baseline ----
    if active("baseline") and can_train:
        banner("6/8 Faz 1 zero-shot taban çizgisi")
        run(["python", "v2/scripts/zero_shot_baseline.py", "--variant", "fill_spatial",
             "--idm-repo", str(args.idm_repo)], gate=False)
        sync_drive(eval_dir)

    # ---- 7. train ----
    if active("train") and can_train:
        banner("7/8 Aşama-1 eğitim (resume'lu)")
        run(["python", "v2/scripts/train_stage1.py", "--max-steps", str(args.train_steps)], gate=True)
        sync_drive(REPO / "v2/checkpoints/stage1")

    # ---- 8. eval ----
    if active("eval") and can_train:
        banner("8/8 değerlendirme")
        preds = eval_dir / "phase1_fill_spatial" / "preds"
        if preds.exists():
            run(["python", "v2/scripts/eval.py", "--pred-dir", str(preds), "--name", "final"], gate=False)
        sync_drive(eval_dir)

    banner("PIPELINE BİTTİ")
    if not can_train:
        print("Not: eğitim aşamaları küçük GPU nedeniyle atlandı — L4/A100 ile tekrar Run all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
