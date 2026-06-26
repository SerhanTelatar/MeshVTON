"""
MeshVTON Phase-2 training — train ControlNet3D on top of the FROZEN real
IDM-VTON pipeline, using IDM-VTON's exact input contract.

Per step (no CFG):
    z0   = vae(person)                                    # target latent
    zt   = add_noise(z0, eps, t)
    unet_in = [zt, mask, vae(masked_image), vae(pose_img)]    # 13ch (IDM-VTON layout)
    cloth_lat        = vae(cloth)            -> GarmentNet -> reference_features
    image_embeds     = encoder_hid_proj(image_encoder(clip(cloth)).hidden[-2])
    residuals        = ControlNet3D(conditioning_3d, t)  # 9 down + 1 mid   (TRAINABLE)
    eps_pred = unet(unet_in, t, text, added_cond_kwargs, down/mid residuals, garment_features)
    loss = MSE(eps_pred, eps)

Only ControlNet3D is trained. Usage:
    python scripts/train_meshvton.py --data_root data --pairs data/raw/train_pairs.csv
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import MeshVTONDataset
from src.models.controlnet_3d import ControlNet3D

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_pipeline(base: str, dtype):
    """Load the frozen real IDM-VTON pipeline components."""
    from src.idm_vton.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.idm_vton.unet_hacked_tryon import UNet2DConditionModel
    from src.idm_vton.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                              CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer)
    from diffusers import DDPMScheduler, AutoencoderKL

    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype)
    unet_enc = UNet2DConditionModel_ref.from_pretrained(base, subfolder="unet_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(base, subfolder="image_encoder", torch_dtype=dtype)
    te1 = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype)
    te2 = CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2", torch_dtype=dtype)
    tok1 = AutoTokenizer.from_pretrained(base, subfolder="tokenizer", use_fast=False)
    tok2 = AutoTokenizer.from_pretrained(base, subfolder="tokenizer_2", use_fast=False)
    scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    pipe = TryonPipeline.from_pretrained(
        base, unet=unet, vae=vae, feature_extractor=CLIPImageProcessor(),
        text_encoder=te1, text_encoder_2=te2, tokenizer=tok1, tokenizer_2=tok2,
        scheduler=scheduler, image_encoder=image_encoder, torch_dtype=dtype)
    pipe.unet_encoder = unet_enc
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="yisol/IDM-VTON")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--pairs", default="data/raw/train_pairs.csv")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.05, help="fraction of steps for LR warmup")
    ap.add_argument("--subset", type=float, default=1.0, help="fraction of dataset")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_dir", default="checkpoints/meshvton")
    ap.add_argument("--save_every", type=int, default=1, help="epochs")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    print("Loading frozen IDM-VTON pipeline...")
    pipe = build_pipeline(args.base, dtype)
    vae, unet, unet_enc = pipe.vae, pipe.unet, pipe.unet_encoder
    image_encoder = pipe.image_encoder
    for m in (vae, unet, unet_enc, image_encoder, pipe.text_encoder, pipe.text_encoder_2):
        m.to(device).requires_grad_(False).eval()
    scaling = vae.config.scaling_factor

    # ControlNet3D — the ONLY trainable module
    controlnet = ControlNet3D(conditioning_channels=9).to(device).train()
    if args.resume:
        controlnet.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed ControlNet3D: {args.resume}")
    n_train = sum(p.numel() for p in controlnet.parameters())
    print(f"Trainable ControlNet3D params: {n_train:,}")

    # Pre-encode the (constant) captions once
    with torch.no_grad():
        pe, _, ppe, _ = pipe.encode_prompt(
            "model is wearing clothes", device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False, negative_prompt="")
        pe_c, _, _, _ = pipe.encode_prompt(
            ["a photo of clothes"], device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False, negative_prompt=[""])
    pe, ppe, pe_c = pe.to(device, dtype), ppe.to(device, dtype), pe_c.to(device, dtype)

    clip_mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    ds = MeshVTONDataset(args.pairs, args.data_root, height=args.height, width=args.width)
    if args.subset < 1.0:
        import random
        idx = random.sample(range(len(ds)), int(len(ds) * args.subset))
        ds = Subset(ds, idx)
    print(f"Dataset: {len(ds)} samples")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)

    opt = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)
    from diffusers.optimization import get_cosine_schedule_with_warmup
    total_steps = len(loader) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    lr_sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"LR schedule: cosine + warmup ({warmup_steps}/{total_steps} steps)")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    num_train_t = pipe.scheduler.config.num_train_timesteps

    def vae_encode(x):
        return vae.encode(x.to(device, dtype)).latent_dist.sample() * scaling

    for epoch in range(args.epochs):
        controlnet.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running = 0.0
        for step, batch in enumerate(pbar):
            b = batch["person"].shape[0]
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                z0 = vae_encode(batch["person"])
                masked_lat = vae_encode(batch["masked_image"])
                pose_lat = vae_encode(batch["pose_img"])
                cloth_lat = vae_encode(batch["cloth"])

                noise = torch.randn_like(z0)
                t = torch.randint(0, num_train_t, (b,), device=device, dtype=torch.long)
                zt = pipe.scheduler.add_noise(z0, noise, t)

                mask_lat = F.interpolate(batch["mask"].to(device, dtype),
                                         size=z0.shape[-2:], mode="nearest")
                unet_in = torch.cat([zt, mask_lat, masked_lat, pose_lat], dim=1)  # 13ch

                # GarmentNet reference features
                _, ref_feats = unet_enc(cloth_lat, t, pe_c.repeat(b, 1, 1), return_dict=False)
                ref_feats = list(ref_feats)

                # IP-Adapter image embeds (trained Resampler in unet.encoder_hid_proj)
                clip_in = (batch["cloth"].to(device, dtype) + 1.0) / 2.0
                clip_in = F.interpolate(clip_in, size=(224, 224), mode="bicubic", align_corners=False)
                clip_in = (clip_in - clip_mean.to(dtype)) / clip_std.to(dtype)
                img_hidden = image_encoder(clip_in, output_hidden_states=True).hidden_states[-2]
                image_embeds = unet.encoder_hid_proj(img_hidden).to(dtype)

                size = [args.height, args.width]
                add_time_ids = torch.tensor([size + [0, 0] + size], device=device, dtype=dtype).repeat(b, 1)
                added = {"text_embeds": ppe.repeat(b, 1), "time_ids": add_time_ids,
                         "image_embeds": image_embeds}

            # ControlNet3D — trainable (gradients flow here)
            with torch.autocast("cuda", dtype=dtype):
                residuals = controlnet(batch["conditioning_3d"].to(device, dtype), t)
                down_res = [r.to(dtype) for r in residuals[:-1]]
                mid_res = residuals[-1].to(dtype)

                eps_pred = unet(
                    unet_in, t, encoder_hidden_states=pe.repeat(b, 1, 1),
                    added_cond_kwargs=added,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                    garment_features=ref_feats, return_dict=False)[0]

                loss = F.mse_loss(eps_pred.float(), noise.float())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            opt.step()
            lr_sched.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{running/(step+1):.4f}",
                             lr=f"{lr_sched.get_last_lr()[0]:.2e}")

        if (epoch + 1) % args.save_every == 0:
            p = save_dir / f"controlnet3d_epoch{epoch+1}.pt"
            torch.save(controlnet.state_dict(), p)
            print(f"Saved {p}")

    torch.save(controlnet.state_dict(), save_dir / "controlnet3d_final.pt")
    print("Training complete.")


if __name__ == "__main__":
    main()
