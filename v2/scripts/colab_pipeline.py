#!/usr/bin/env python3
"""Colab TEK-KOŞU pipeline'ı — bakıcılık istemez; her aşama idempotent/resume'ludur.

Aşamalar (sırayla; biten aşama tekrar koşulmaz):
  1. check     : GPU/VRAM/ortam kontrolü (eğitim için >=20GB ister; yoksa veri-only mod)
  2. camera    : Faz 2 kapısı — reprojection IoU >= 0.70 (KALIRSA DURUR)
  3. smoke     : sentetik duman testi (5 örnek) + QA kontak-sheet'i Drive'a
  4. synth     : tam sentetik üretim (--num, varsayılan 2000; 4 görüş/örnek)
  5. baseline  : Faz 1 zero-shot taban çizgisi (fill_spatial, tam golden set)
  6. train     : Aşama-1 eğitim (resume'lu; Colab koparsa tekrar Run all yeter)
  7. eval      : golden set değerlendirme + kontrol-ablation kapısı

NOT: VITON-HD gerçek-foto yolu (preprocess_vitonhd.py) bilinçli olarak pipeline'a
DAHİL DEĞİL — appearance ref orada mesh yok, tek sinyal renkli ürün fotoğrafı/kırpım,
bu da projenin KALICI texturesuz kuralına aykırı düşer. Script tooling olarak duruyor,
ama varsayılan eğitim akışı sadece sentetik (texture'sız-gri) veriyle çalışır.

Kullanım (notebook tek hücre):
  python v2/scripts/colab_pipeline.py --idm-repo /content/IDM-VTON \\
      [--until train] [--num 2000] [--skip baseline]

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
STAGES = ("check", "camera", "smoke", "synth", "baseline", "train", "eval")


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n== {msg}\n{'=' * 70}", flush=True)


def run(cmd: list[str], gate: bool = False) -> int:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.call([sys.executable, *cmd[1:]] if cmd[0] == "python" else cmd)
    if rc != 0 and gate:
        raise SystemExit(f"KAPI KALDI (rc={rc}): {' '.join(map(str, cmd))} — pipeline durdu.")
    return rc


def archive_to_drive(src_dir: Path, zip_name: str) -> None:
    """Üretilen veri setini Drive'a STORE zip'i olarak arşivle (PNG zaten sıkışık,
    hız için sıkıştırma yok). Colab diski oturumla ölür — bu, sonraki oturumun
    veriyi yeniden üretmemesinin tek yolu."""
    if not DRIVE_OUT.parent.parent.exists() or not src_dir.exists():
        return
    import zipfile

    DRIVE_OUT.mkdir(parents=True, exist_ok=True)
    dst = DRIVE_OUT / zip_name
    tmp = dst.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for f in src_dir.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(src_dir.parent))
    tmp.rename(dst)
    print(f"→ arşivlendi: {dst} ({dst.stat().st_size / 1e9:.1f} GB)", flush=True)


def restore_from_drive(zip_name: str, target_parent: Path) -> bool:
    zpath = DRIVE_OUT / zip_name
    if not zpath.exists():
        return False
    import zipfile

    print(f"Drive arşivinden geri yükleniyor: {zpath}", flush=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(target_parent)
    return True


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
    ap.add_argument("--poses", type=Path, default=REPO / "data/smplx_params")
    ap.add_argument("--num", type=int, default=2000, help="sentetik örnek sayısı")
    ap.add_argument("--train-steps", type=int, default=20000)
    ap.add_argument("--until", choices=STAGES, default="eval")
    ap.add_argument("--skip", action="append", default=[], choices=STAGES)
    args = ap.parse_args()

    until = STAGES.index(args.until)
    eval_dir = REPO / "v2/eval_results"
    synth_dir = REPO / "v2/data/synth"
    # train_stage1.py'nin varsayılanıyla eşleşir (stage1_singleview.yaml: train.out_dir);
    # Drive mount edilmişse checkpoint doğrudan oraya (oturum kopmasına karşı), değilse yerel.
    stage1_out_dir = (DRIVE_OUT / "stage1") if DRIVE_OUT.parent.parent.exists() \
        else (REPO / "v2/checkpoints/stage1")
    can_train = True

    def active(stage: str) -> bool:
        return STAGES.index(stage) <= until and stage not in args.skip

    def purge_stale_synth(where: str) -> None:
        """Sürümü eskiyen sentetik veriyi (yerel + Drive arşivi) siler.

        DUMAN AŞAMASINDAN ÖNCE de çağrılır: kontrol eskiden yalnız 4/7'deydi, ama
        3/7 "diskte örnek var" deyip üretimi atlıyordu → --until smoke ile koşan
        kullanıcı QA'da BAYAT görsel görüyordu (düzeltmelerin etkisi görünmüyordu,
        2026-08-09). Sürüm kaynaktan okunur, elle yazılmaz.
        """
        sys.path.insert(0, str(REPO / "v2"))
        from meshvton2.synth.generate import DATA_VERSION

        if not synth_dir.exists() or not any(synth_dir.glob("s*_*/")):
            return
        vf = synth_dir / "DATA_VERSION"
        if vf.exists() and vf.read_text().strip() == DATA_VERSION:
            return
        print(f"UYARI [{where}]: sentetik veri ESKİ SÜRÜM (beklenen v{DATA_VERSION}) — "
              "siliniyor ve sıfırdan üretilecek.")
        shutil.rmtree(synth_dir, ignore_errors=True)
        (DRIVE_OUT / "synth_data.zip").unlink(missing_ok=True)

    # ---- 1. check ----
    banner("1/7 ortam kontrolü")
    try:
        head = subprocess.check_output(["git", "-C", str(REPO), "log", "--oneline", "-1"], text=True).strip()
        print(f"repo: {head}")  # düzeltmelerin gelip gelmediği buradan doğrulanır
    except Exception:
        pass
    name, vram = gpu_info()
    print(f"GPU: {name} ({vram:.0f} GB)")
    if name == "YOK":
        raise SystemExit(
            "HATA: GPU yok (CPU runtime). Runtime -> Change runtime type -> GPU seçin.\n"
            "Prova (veri+doğrulama) için T4 yeter; eğitim için L4/A100."
        )
    if vram < 30:  # L4 (24GB) FLUX eğitimine yetmez (ağırlıklar tek başına ~24GB) → OOM önlemi
        can_train = False
        print("UYARI: <30GB VRAM — FLUX aşamaları (baseline/train) ATLANACAK; "
              "veri üretimi + doğrulama yine koşar. Eğitim için A100 runtime seçin.")
    # bağımlılık denetimi: eksikler tek tek, kurulum ipucuyla (kapıda sürpriz yerine burada dur)
    deps = {
        "pyrender": "pip install pyrender", "smplx": "pip install smplx",
        "trimesh": "pip install trimesh", "onnxruntime": "pip install onnxruntime",
        "hmr2": 'pip install "git+https://github.com/shubham-goel/4D-Humans.git"',
        "diffusers": 'pip install "diffusers>=0.34"', "peft": 'pip install "peft>=0.14"',
    }
    missing = []
    import importlib.util as _ilu

    for mod, hint in deps.items():
        if _ilu.find_spec(mod) is None:
            missing.append(f"  {mod}  ->  {hint}")
    if missing:
        raise SystemExit("HATA: eksik bağımlılıklar (kurulum hücresi koşmadı mı?):\n" + "\n".join(missing))
    print("bağımlılıklar tamam")

    # ---- 2. camera gate ----
    if active("camera"):
        banner("2/7 kamera doğrulama kapısı (IoU >= 0.70)")
        if not (REPO / "v2/data/golden/manifest.json").exists():
            run(["python", "v2/scripts/build_golden_set.py",
                 "--vitonhd-test", str(args.images), "--garments", str(args.garments)], gate=True)
        run(["python", "v2/scripts/validate_camera.py", "--idm-repo", str(args.idm_repo), "--limit", "5"],
            gate=True)
        sync_drive(eval_dir / "camera_validation")

    # ---- 3. smoke + QA ----
    if active("smoke"):
        banner("3/7 sentetik duman testi + QA sheet")
        purge_stale_synth("3/7")  # QA görseli TAZE kodla üretilsin (bayat QA tuzağı)
        # pairs.csv değil ÖRNEK sayısına bak: başarısız koşu header'lı boş csv bırakabilir
        n_existing = sum(1 for _ in synth_dir.glob("s*_*/")) if synth_dir.exists() else 0
        if n_existing:
            print(f"NOT: diskte {n_existing} mevcut örnek var — duman üretimi ATLANIYOR. "
                  "Taze duman istiyorsanız v2/data/synth silinmeli.")
        if not n_existing:
            run(["python", "v2/scripts/generate_synthetic.py", "--garments", str(args.garments),
                 "--poses", str(args.poses), "--num", "5", "--limit-garments", "3"], gate=True)
        if not any(synth_dir.glob("s*_*/")):
            raise SystemExit("KAPI: duman testinde hiç örnek yazılamadı — drape hattını inceleyin")
        qa = REPO / "v2/eval_results/synth_qa"
        qa.mkdir(parents=True, exist_ok=True)
        first = sorted(synth_dir.glob("s*_*/"))
        if first:
            shutil.copytree(first[0], qa / first[0].name, dirs_exist_ok=True)
        sync_drive(qa)
        print("QA görselleri Drive'da: v2_outputs/synth_qa — view_180/gt.png'yi GÖZLE kontrol edin.")

    # ---- 4. full synth ----
    if active("synth"):
        banner(f"4/7 tam sentetik üretim (hedef {args.num} YAZILMIŞ örnek × 4 görüş)")
        count = lambda: sum(1 for _ in synth_dir.glob("s*_*/")) if synth_dir.exists() else 0

        if count() < args.num:
            restore_from_drive("synth_data.zip", synth_dir.parent)
        purge_stale_synth("4/7")  # Drive arşivinden GELEN eski sürümü de yakalar
        existing = count()
        # PARÇALI + PARALEL üretim: her parça sonrası Drive arşivi (kopma en fazla
        # bir parça götürür); çok çekirdekli makinede (A100 VM ~12 vCPU) işçiler
        # paralel — üretim CPU-bound, GPU değil.
        import multiprocessing

        workers = 4 if multiprocessing.cpu_count() >= 8 else 1
        chunk = 200 * workers
        print(f"paralel işçi: {workers} (cpu={multiprocessing.cpu_count()})")
        salt = 0  # tekrar denemelerde seed'ler TAZE olsun (aynı kimlik → aynı RED döngüsü olmasın)
        while existing < args.num:
            remaining = args.num - existing
            w = min(workers, remaining)
            per = max(1, min(chunk, remaining) // w)
            cmds = [
                [sys.executable, "v2/scripts/generate_synthetic.py", "--garments", str(args.garments),
                 "--poses", str(args.poses), "--num", str(per), "--seed", str(existing + salt + k)]
                for k in range(w)
            ]
            procs = [subprocess.Popen(c) for c in cmds]
            rcs = [p.wait() for p in procs]
            salt += w
            new = count()
            archive_to_drive(synth_dir, "synth_data.zip")
            if new == existing:  # bu turda hiç YENİ örnek yok
                if remaining <= 2 * workers:
                    # kuyruk: kalan birkaç örnek red şansına takılıyor — hedefe yeterince yakınız
                    print(f"NOT: son {remaining} örnek red'lere takıldı; {new} örnekle devam (hedefe ~%100 yakın)")
                    break
                raise SystemExit(f"HATA: parça hiç örnek üretemedi (rc={rcs}) — drape red/hata oranını inceleyin")
            print(f"ilerleme: {new}/{args.num} yazılmış örnek")
            existing = new
        print(f"sentetik veri hazır: {existing} örnek")
        sync_drive(synth_dir / "pairs.csv")

    # ---- 5. baseline ----
    if active("baseline") and can_train:
        banner("5/7 Faz 1 zero-shot taban çizgisi")
        done_marker = (eval_dir / "phase1_fill_spatial.json",
                       DRIVE_OUT / "eval_results" / "phase1_fill_spatial.json")
        if any(m.exists() for m in done_marker):
            print("baseline raporu zaten var (lokal/Drive) — atlanıyor")
        else:
            run(["python", "v2/scripts/zero_shot_baseline.py", "--variant", "fill_spatial",
                 "--idm-repo", str(args.idm_repo)], gate=False)
            sync_drive(eval_dir)

    # ---- 6. train ----
    if active("train") and can_train:
        banner("6/7 Aşama-1 eğitim (resume'lu)")
        # Latent ön-hesabı (bir kez ~20-30dk): adımdan VAE+PNG çıkar → ~1.5-2x hız,
        # sonuç bit-eş (aynı VAE) — resume-safe, mevcutları atlar
        run(["python", "v2/scripts/precompute_latents.py"], gate=True)
        # checkpoint'ler DOĞRUDAN Drive'a: 20k adım tek oturuma sığmayabilir;
        # kopma anında son ckpt kaybolmasın (latest.pt oradan resume eder)
        run(["python", "v2/scripts/train_stage1.py", "--max-steps", str(args.train_steps),
             "--out-dir", str(stage1_out_dir)], gate=True)

    # ---- 7. eval ----
    if active("eval") and can_train:
        banner("7/7 değerlendirme (eğitilmiş checkpoint + kontrol-ablation kapısı)")
        # final.pt yalnız eğitim max-steps'e ULAŞIP bitince yazılır (train_stage1.py).
        # Eğitim yarıda kesildiyse (--skip train ile buraya atlandıysa) final.pt yok —
        # bu durumda TrainLoop.save()'in yazdığı "latest.pt" işaretçisini çözüp
        # en son periyodik checkpoint'i (ckpt_XXXXXX.pt) kullan (kısmi ilerleme de
        # değerlendirilebilsin).
        final_ckpt = stage1_out_dir / "final.pt"
        latest_ptr = stage1_out_dir / "latest.pt"
        if final_ckpt.exists():
            ckpt = final_ckpt
        elif latest_ptr.exists():
            ckpt = Path(latest_ptr.read_text().strip())
            print(f"NOT: final.pt yok — kısmi checkpoint kullanılıyor: {ckpt}")
        else:
            ckpt = None
        if ckpt and ckpt.exists():
            # eval_checkpoint.py kendi içinde control_on/control_off üretip geo_iou
            # kapısını koşar (ckpt_control_on.json / ckpt_control_off.json yazar) —
            # eskiden burada 5. aşamanın zero-shot preds'i tekrar değerlendiriliyordu,
            # eğitimin gerçek çıktısı hiç ölçülmüyordu.
            run(["python", "v2/scripts/eval_checkpoint.py", "--checkpoint", str(ckpt),
                 "--idm-repo", str(args.idm_repo)], gate=False)
        else:
            print(f"UYARI: checkpoint yok ({stage1_out_dir}) — eval atlanıyor.")
        sync_drive(eval_dir)

    banner("PIPELINE BİTTİ")
    if not can_train:
        print("Not: eğitim aşamaları küçük GPU nedeniyle atlandı — L4/A100 ile tekrar Run all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
