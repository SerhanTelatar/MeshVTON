#!/usr/bin/env python3
"""Colab SINGLE-RUN pipeline — no babysitting needed; every stage is idempotent/resumable.

Stages (in order; a completed stage is not re-run):
  1. check     : GPU/VRAM/environment check (needs >=20GB for training; otherwise data-only mode)
  2. camera    : Phase 2 gate — reprojection IoU >= 0.70 (STOPS IF IT FAILS)
  3. smoke     : synthetic smoke test (5 samples) + a QA contact sheet to Drive
  4. synth     : full synthetic generation (--num, default 2000; 4 views/sample)
  5. baseline  : Phase 1 zero-shot baseline (fill_spatial, full golden set)
  6. train     : Stage-1 training (resumable; if Colab drops, another Run all is enough)
  7. eval      : golden set evaluation + control-ablation gate

NOTE: the VITON-HD real-photo path (preprocess_vitonhd.py) is deliberately NOT part of
the pipeline — there is no mesh there for the appearance ref, the only signal is a color
product photo/crop, which conflicts with the project's PERMANENT textureless rule. The script
remains as tooling, but the default training flow runs on synthetic (textureless grey) data only.

Usage (a single notebook cell):
  python v2/scripts/colab_pipeline.py --idm-repo /content/IDM-VTON \\
      [--until train] [--num 2000] [--skip baseline]

After each stage the critical outputs are copied to Drive (if it is mounted):
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
        raise SystemExit(f"GATE FAILED (rc={rc}): {' '.join(map(str, cmd))} — pipeline stopped.")
    return rc


def archive_to_drive(src_dir: Path, zip_name: str) -> None:
    """Archive the generated dataset to Drive as a STORE zip (PNGs are already compressed,
    no compression, for speed). The Colab disk dies with the session — this is the only way
    to keep the next session from regenerating the data."""
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
    print(f"→ archived: {dst} ({dst.stat().st_size / 1e9:.1f} GB)", flush=True)


def restore_from_drive(zip_name: str, target_parent: Path) -> bool:
    zpath = DRIVE_OUT / zip_name
    if not zpath.exists():
        return False
    import zipfile

    print(f"Restoring from the Drive archive: {zpath}", flush=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(target_parent)
    return True


def sync_drive(*paths: Path) -> None:
    if not DRIVE_OUT.parent.parent.exists():  # drive is not mounted
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
    print(f"→ copied to Drive: {DRIVE_OUT}", flush=True)


def gpu_info() -> tuple[str, float]:
    import torch

    if not torch.cuda.is_available():
        return "NONE", 0.0
    p = torch.cuda.get_device_properties(0)
    return p.name, p.total_memory / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idm-repo", type=Path, default=Path("/content/IDM-VTON"))
    ap.add_argument("--garments", type=Path, default=REPO / "data/garments_3d")
    ap.add_argument("--images", type=Path, default=REPO / "data/raw/images")
    ap.add_argument("--poses", type=Path, default=REPO / "data/smplx_params")
    ap.add_argument("--num", type=int, default=2000, help="number of synthetic samples")
    ap.add_argument("--train-steps", type=int, default=20000)
    ap.add_argument("--until", choices=STAGES, default="eval")
    ap.add_argument("--skip", action="append", default=[], choices=STAGES)
    args = ap.parse_args()

    until = STAGES.index(args.until)
    eval_dir = REPO / "v2/eval_results"
    synth_dir = REPO / "v2/data/synth"
    # Matches train_stage1.py's default (stage1_singleview.yaml: train.out_dir);
    # if Drive is mounted the checkpoints go straight there (against session drops), otherwise local.
    stage1_out_dir = (DRIVE_OUT / "stage1") if DRIVE_OUT.parent.parent.exists() \
        else (REPO / "v2/checkpoints/stage1")
    can_train = True

    def active(stage: str) -> bool:
        return STAGES.index(stage) <= until and stage not in args.skip

    def purge_stale_synth(where: str) -> None:
        """Deletes stale-version synthetic data (local + the Drive archive).

        Also called BEFORE the SMOKE STAGE: the check used to live only in 4/7, but
        3/7 said "there are samples on disk" and skipped generation → a user running
        with --until smoke saw STALE images in QA (the effect of the fixes was invisible,
        2026-08-09). The version is read from the source, never hand-written.
        """
        sys.path.insert(0, str(REPO / "v2"))
        from meshvton2.synth.generate import DATA_VERSION

        if not synth_dir.exists() or not any(synth_dir.glob("s*_*/")):
            return
        vf = synth_dir / "DATA_VERSION"
        if vf.exists() and vf.read_text().strip() == DATA_VERSION:
            return
        print(f"WARNING [{where}]: synthetic data is an OLD VERSION (expected v{DATA_VERSION}) — "
              "deleting it and regenerating from scratch.")
        shutil.rmtree(synth_dir, ignore_errors=True)
        (DRIVE_OUT / "synth_data.zip").unlink(missing_ok=True)

    # ---- 1. check ----
    banner("1/7 environment check")
    try:
        head = subprocess.check_output(["git", "-C", str(REPO), "log", "--oneline", "-1"], text=True).strip()
        print(f"repo: {head}")  # this is how you confirm whether the fixes arrived
    except Exception:
        pass
    name, vram = gpu_info()
    print(f"GPU: {name} ({vram:.0f} GB)")
    if name == "NONE":
        raise SystemExit(
            "ERROR: no GPU (CPU runtime). Runtime -> Change runtime type -> pick GPU.\n"
            "A T4 is enough for a dry run (data+validation); training needs an L4/A100."
        )
    if vram < 30:  # an L4 (24GB) cannot train FLUX (the weights alone are ~24GB) → OOM guard
        can_train = False
        print("WARNING: <30GB VRAM — the FLUX stages (baseline/train) WILL BE SKIPPED; "
              "data generation + validation still run. Pick an A100 runtime for training.")
    # dependency check: missing ones listed one by one with an install hint (stop here instead of at the gate)
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
        raise SystemExit("ERROR: missing dependencies (did the install cell not run?):\n" + "\n".join(missing))
    print("dependencies ok")

    # ---- 2. camera gate ----
    if active("camera"):
        banner("2/7 camera validation gate (IoU >= 0.70)")
        if not (REPO / "v2/data/golden/manifest.json").exists():
            run(["python", "v2/scripts/build_golden_set.py",
                 "--vitonhd-test", str(args.images), "--garments", str(args.garments)], gate=True)
        run(["python", "v2/scripts/validate_camera.py", "--idm-repo", str(args.idm_repo), "--limit", "5"],
            gate=True)
        sync_drive(eval_dir / "camera_validation")

    # ---- 3. smoke + QA ----
    if active("smoke"):
        banner("3/7 synthetic smoke test + QA sheet")
        purge_stale_synth("3/7")  # make sure the QA image is produced with FRESH code (the stale-QA trap)
        # Look at the SAMPLE count, not pairs.csv: a failed run can leave an empty csv with a header
        n_existing = sum(1 for _ in synth_dir.glob("s*_*/")) if synth_dir.exists() else 0
        if n_existing:
            print(f"NOTE: {n_existing} existing samples on disk — SKIPPING smoke generation. "
                  "Delete v2/data/synth if you want a fresh smoke run.")
        if not n_existing:
            run(["python", "v2/scripts/generate_synthetic.py", "--garments", str(args.garments),
                 # 8 DIFFERENT garments: QA cannot be judged from a single mesh — a broken
                 # CLOTH3D mesh (split panels, holes) was being mistaken for the whole pipeline
                 # (2026-08-09: QA kept showing 00047_Top).
                 "--poses", str(args.poses), "--num", "8", "--limit-garments", "8"], gate=True)
        if not any(synth_dir.glob("s*_*/")):
            raise SystemExit("GATE: the smoke test wrote no samples at all — inspect the drape pipeline")
        qa = REPO / "v2/eval_results/synth_qa"
        shutil.rmtree(qa, ignore_errors=True)  # keep the previous round's images from mixing in
        qa.mkdir(parents=True, exist_ok=True)
        samples = sorted(synth_dir.glob("s*_*/"))[:6]  # a single sample is not a statistic
        for sd in samples:
            shutil.copytree(sd, qa / sd.name, dirs_exist_ok=True)
        sync_drive(qa)
        print(f"QA: {len(samples)} DIFFERENT garment samples on Drive (v2_outputs/synth_qa).")
        print("LOOK at them: does the garment sit on the shoulder, does the body poke through the cloth, "
              "are 0° and 180° consistent. One broken mesh is normal — judge the MAJORITY.")

    # ---- 4. full synth ----
    if active("synth"):
        banner(f"4/7 full synthetic generation (target {args.num} WRITTEN samples × 4 views)")
        count = lambda: sum(1 for _ in synth_dir.glob("s*_*/")) if synth_dir.exists() else 0

        if count() < args.num:
            restore_from_drive("synth_data.zip", synth_dir.parent)
        purge_stale_synth("4/7")  # also catches an old version RESTORED from the Drive archive
        existing = count()
        # CHUNKED + PARALLEL generation: a Drive archive after every chunk (a drop costs
        # at most one chunk); on a many-core machine (A100 VM ~12 vCPU) the workers run
        # in parallel — generation is CPU-bound, not GPU-bound.
        import multiprocessing

        workers = 4 if multiprocessing.cpu_count() >= 8 else 1
        chunk = 200 * workers
        print(f"parallel workers: {workers} (cpu={multiprocessing.cpu_count()})")
        salt = 0  # keep seeds FRESH across retries (same identity → same REJECT loop)
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
            if new == existing:  # no NEW samples at all this round
                if remaining <= 2 * workers:
                    # tail: the last few samples keep hitting rejections — close enough to the target
                    print(f"NOTE: the last {remaining} samples hit rejections; continuing with {new} samples (~100% of target)")
                    break
                raise SystemExit(f"ERROR: the chunk produced no samples at all (rc={rcs}) — inspect the drape reject/error rate")
            print(f"progress: {new}/{args.num} samples written")
            existing = new
        print(f"synthetic data ready: {existing} samples")
        sync_drive(synth_dir / "pairs.csv")

    # ---- 5. baseline ----
    if active("baseline") and can_train:
        banner("5/7 Phase 1 zero-shot baseline")
        done_marker = (eval_dir / "phase1_fill_spatial.json",
                       DRIVE_OUT / "eval_results" / "phase1_fill_spatial.json")
        if any(m.exists() for m in done_marker):
            print("the baseline report already exists (local/Drive) — skipping")
        else:
            run(["python", "v2/scripts/zero_shot_baseline.py", "--variant", "fill_spatial",
                 "--idm-repo", str(args.idm_repo)], gate=False)
            sync_drive(eval_dir)

    # ---- 6. train ----
    if active("train") and can_train:
        banner("6/7 Stage-1 training (resumable)")
        # Latent precompute (~20-30 min, once): takes VAE+PNG out of the step → ~1.5-2x faster,
        # bit-identical results (same VAE) — resume-safe, skips existing ones
        run(["python", "v2/scripts/precompute_latents.py"], gate=True)
        # checkpoints go DIRECTLY to Drive: 20k steps may not fit in one session;
        # the last ckpt must survive a drop (latest.pt resumes from there)
        run(["python", "v2/scripts/train_stage1.py", "--max-steps", str(args.train_steps),
             "--out-dir", str(stage1_out_dir)], gate=True)

    # ---- 7. eval ----
    if active("eval") and can_train:
        banner("7/7 evaluation (trained checkpoint + control-ablation gate)")
        # final.pt is only written when training REACHES max-steps and finishes (train_stage1.py).
        # If training was cut short (or we jumped here with --skip train) there is no final.pt —
        # in that case resolve the "latest.pt" pointer written by TrainLoop.save() and use
        # the most recent periodic checkpoint (ckpt_XXXXXX.pt), so partial progress can
        # be evaluated too.
        final_ckpt = stage1_out_dir / "final.pt"
        latest_ptr = stage1_out_dir / "latest.pt"
        if final_ckpt.exists():
            ckpt = final_ckpt
        elif latest_ptr.exists():
            ckpt = Path(latest_ptr.read_text().strip())
            print(f"NOTE: no final.pt — using a partial checkpoint: {ckpt}")
        else:
            ckpt = None
        if ckpt and ckpt.exists():
            # eval_checkpoint.py generates control_on/control_off itself and runs the
            # geo_iou gate (writing ckpt_control_on.json / ckpt_control_off.json) —
            # this used to re-evaluate stage 5's zero-shot preds, so the training's
            # actual output was never measured.
            run(["python", "v2/scripts/eval_checkpoint.py", "--checkpoint", str(ckpt),
                 "--idm-repo", str(args.idm_repo)], gate=False)
        else:
            print(f"WARNING: no checkpoint ({stage1_out_dir}) — skipping eval.")
        sync_drive(eval_dir)

    banner("PIPELINE FINISHED")
    if not can_train:
        print("Note: training stages were skipped because of the small GPU — Run all again with an L4/A100.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
