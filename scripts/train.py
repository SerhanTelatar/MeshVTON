"""Training entry point: python scripts/train.py --config configs/train.yaml"""

import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.models.tryon_pipeline import TryOnPipeline
from src.data.dataset import TryOnDataset
from src.data.transforms import TryOnTransforms
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="DiffFit-3D Training")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local", action="store_true",
                        help="Use placeholder models (no HuggingFace download)")
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    # Set seed
    seed = args.seed or config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Config: {args.config}")
    print(f"Seed: {seed}")

    # Build model
    model_cfg = config.get("model", {})

    if args.local or not model_cfg.get("pretrained_idm_vton"):
        # Local/dev mode — no HuggingFace download
        print("Building pipeline with placeholder modules (local mode)")
        pipeline = TryOnPipeline.from_config(model_cfg)
    else:
        # Production mode — load IDM-VTON from HuggingFace
        pipeline = TryOnPipeline.from_pretrained(
            pretrained_model_id=model_cfg["pretrained_idm_vton"],
            controlnet_3d_channels=model_cfg.get("controlnet_3d_channels", 9),
        )

    # Parameter counts
    total_params = sum(p.numel() for p in pipeline.parameters())
    trainable_params = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\nModel Summary:")
    print(f"  Total parameters:     {total_params:>15,}")
    print(f"  Trainable (ControlNet3D): {trainable_params:>11,}")
    print(f"  Frozen (IDM-VTON):    {frozen_params:>15,}")
    print(f"  Trainable ratio:      {100 * trainable_params / max(total_params, 1):>14.1f}%\n")

    # Build dataset
    data_cfg = config.get("data", {})
    aug_cfg = OmegaConf.load(data_cfg.get("config", "configs/data/dataset.yaml"))
    aug_cfg = OmegaConf.to_container(aug_cfg, resolve=True)

    transforms = TryOnTransforms(aug_cfg.get("augmentation", {}))
    train_dataset = TryOnDataset(
        pairs_file=aug_cfg.get("train_pairs", "data/raw/train_pairs.csv"),
        data_root=aug_cfg.get("data_root", "data"),
        resolution=aug_cfg.get("image", {}).get("resolution", 512),
        transforms=transforms,
    )

    from torch.utils.data import Subset
    import random

    subset_ratio = 0.1  # Use 10% of the dataset
    total = len(train_dataset)
    indices = random.sample(range(total), int(total * subset_ratio))
    train_dataset = Subset(train_dataset, indices)
    print(f"Subset: using {len(train_dataset)}/{total} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("persistent_workers", True) if data_cfg.get("num_workers", 2) > 0 else False,
        drop_last=True,
    )

    # Validation dataset
    val_loader = None
    val_pairs = aug_cfg.get("val_pairs", "data/raw/val_pairs.csv")
    if Path(val_pairs).exists():
        val_dataset = TryOnDataset(
            pairs_file=val_pairs, data_root=aug_cfg.get("data_root", "data"),
            resolution=aug_cfg.get("image", {}).get("resolution", 512),
        )
        val_loader = DataLoader(val_dataset, batch_size=data_cfg.get("batch_size", 1), shuffle=False, num_workers=2)

    # Build trainer
    trainer = Trainer(pipeline, train_loader, val_loader, config)

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)
    else:
        import glob
        save_dir = config.get("training", {}).get("checkpoint", {}).get("save_dir", "checkpoints/runs")
        checkpoints = sorted(glob.glob(f"{save_dir}/*.pt"))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"Auto-resuming from: {latest}")
            trainer.load_checkpoint(latest)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
