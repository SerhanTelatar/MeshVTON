#!/usr/bin/env python3
"""Aşama 1 eğitimi — tek görüş (Colab, büyük GPU: L4/A100; T4'e sığmaz).

Kullanım:
  python v2/scripts/train_stage1.py                     # config varsayılanları
  python v2/scripts/train_stage1.py --smoke             # 10 adımlık duman testi
  python v2/scripts/train_stage1.py --max-steps 5000    # override

Veri önkoşulları:
  - sentetik: v2/data/synth (generate_synthetic.py)
  - gerçek (opsiyonel ama önerilir): v2/data/vitonhd_items (preprocess_vitonhd.py)
Resume otomatiktir (out_dir/latest.pt varsa kaldığı adımdan devam eder).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "v2"))

import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from meshvton2.data.datasets import MixedDataset, SingleViewDataset  # noqa: E402
from meshvton2.data.items import discover_flat_items, discover_synth_items  # noqa: E402
from meshvton2.model.flux_tryon import FluxTrainModule  # noqa: E402
from meshvton2.training.loop import TrainConfig, TrainLoop  # noqa: E402


def build_dataset(cfg: dict, size: tuple[int, int]):
    synth_root = REPO / cfg["data"]["synth_root"]
    real_root = REPO / cfg["data"]["real_root"]
    synth_items = discover_synth_items(synth_root)
    # latent önbelleği: TÜM öğelerde varsa VAE'siz hızlı yol (precompute_latents.py)
    use_latents = all((it.item_dir / "latents.pt").exists() for it in synth_items[:50]) and \
        (synth_items[0].item_dir / "latents.pt").exists()
    synth = SingleViewDataset(synth_items, size=size, use_latents=use_latents)
    print(f"sentetik: {len(synth)} öğe ({synth_root}); latent önbelleği: {'AÇIK' if use_latents else 'kapalı'}")
    if real_root.exists():
        real_items = discover_flat_items(real_root)
        if use_latents and not (real_items[0].item_dir / "latents.pt").exists():
            use_latents = False  # karışık mod olmaz; ikisi de latent'li olmalı
            synth = SingleViewDataset(synth_items, size=size, use_latents=False)
            print("NOT: gerçek veride latent yok — iki taraf da piksel modunda")
        real = SingleViewDataset(real_items, size=size, use_latents=use_latents)
        print(f"gerçek: {len(real)} öğe; karışım oranı {cfg['data']['real_ratio']}")
        return MixedDataset(real, synth, primary_ratio=cfg["data"]["real_ratio"])
    print("UYARI: gerçek veri yok — sentetik-only eğitim (domain-gap riski; "
          "preprocess_vitonhd.py koşulması önerilir)", file=sys.stderr)
    return synth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO / "v2/configs/stage1_singleview.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="checkpoint dizini (örn. Drive yolu)")
    ap.add_argument("--smoke", action="store_true", help="10 adım, ckpt yok — kurulum doğrulama")
    args = ap.parse_args()

    base = yaml.safe_load((REPO / "v2/configs/base.yaml").read_text())
    cfg = yaml.safe_load(args.config.read_text())
    size = (base["resolution"]["height"], base["resolution"]["width"])

    ds = build_dataset(cfg, size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size or cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )

    module = FluxTrainModule(
        base["model"]["flux_fill_repo"],
        prompt=base["model"]["prompt"],
        lora_rank=cfg["model"]["lora_rank"],
        lora_alpha=cfg["model"]["lora_alpha"],
        ref_dropout=cfg["model"]["ref_dropout"],
        train_guidance=cfg["model"]["train_guidance"],
        compile_transformer=cfg["model"].get("compile", False),
    ).setup()

    t = cfg["train"]
    train_cfg = TrainConfig(
        max_steps=10 if args.smoke else (args.max_steps or t["max_steps"]),
        lr=t["lr"], weight_decay=t["weight_decay"], warmup_steps=0 if args.smoke else t["warmup_steps"],
        grad_accum=1 if args.smoke else t["grad_accum"], max_grad_norm=t["max_grad_norm"],
        ckpt_every=10 ** 9 if args.smoke else t["ckpt_every"], log_every=1 if args.smoke else t["log_every"],
        keep_last=t.get("keep_last", 2),
        out_dir=str(args.out_dir or REPO / t["out_dir"]),
    )
    loop = TrainLoop(
        module.trainable_parameters(), module.step, train_cfg,
        state_provider=module.trainable_state, state_loader=module.load_trainable_state,
    )
    if not args.smoke:
        loop.maybe_resume()
    loop.run(loader)
    if not args.smoke:
        loop.save(Path(train_cfg.out_dir) / "final.pt")
        print(f"bitti — checkpoint: {train_cfg.out_dir}/final.pt")
    else:
        print("duman testi OK (10 adım, loss yukarıda)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
