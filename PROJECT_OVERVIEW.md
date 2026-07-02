# MeshVTON — Project Overview

> **Multi-View Virtual Try-On via 3D Garment Mesh Conditioning in Latent Diffusion Models**
>
> This document summarizes the entire MeshVTON project end to end — its purpose, architecture, data
> flow, training/inference strategy, runtime environment, and current status. It is based on the
> codebase, the memory notes, and the existing documentation.

---

## 1. The Project in One Sentence

MeshVTON is a virtual try-on pipeline that takes a **2D human photo** and a **3D garment mesh
(`.obj`)** as input and produces the garment fitted to that person's estimated pose, in a way that
is **geometrically consistent from any camera angle (front / side / back)**. The output is a 2D
photorealistic image of the person wearing the garment.

Its core carrier is the pretrained **IDM-VTON (SDXL-based Latent Diffusion)** model. The project's
**novelty** is the **ControlNet3D** module, which injects 3D geometric conditioning into this frozen
backbone.

---

## 2. Why Does It Exist? (Motivation and Novelty)

Classic 2D virtual try-on methods rely on a single flat garment photo and suffer from these
problems:

- **Front-facing bias** — Since the network only sees the garment from the front, it keeps drawing
  the front of the garment even when the person turns around.
- **Perspective inconsistency** — Side/back views look artificial, like a "sticker".
- **Fragility to body type** — 2D warping cannot generalize to different body types.

MeshVTON solves this using **real 3D garment meshes**:

| Advantage | Description |
|---------|----------|
| 🔄 Back & side views | The mesh is rendered from any angle; no hallucination needed |
| 📐 Geometric accuracy | The 3D mesh guarantees correct proportions on every body type |
| 🎭 Normal & depth maps | Provides 3D structural cues to the diffusion model via ControlNet3D |
| 💡 Physical lighting | Phong shading produces realistic shadow/highlight |
| 🧠 IDM-VTON backbone | The pretrained SDXL try-on model provides SOTA generation quality |

**Paradigm comparison:**

```
Classic 2D try-on:  2D garment photo → warp → paste → result (front only)

MeshVTON:          3D garment mesh → SMPL-X draping → PyTorch3D render
                          │                                       │
                   GarmentNet (frozen)                ControlNet3D (trainable)
                          │                                       │
                          └────►  TryonNet (frozen)  ◄────────────┘
                                         │
                                  VAE decode → result (from any angle)
```

---

## 3. High-Level Pipeline

| Stage | Component | Description |
|-------|---------|----------|
| **Input** | Human image + 3D garment mesh | 2D photo + `.obj` garment file |
| **Body Estimation** | SMPL-X Estimator | 3D body shape (β), pose (θ), joints |
| **Garment Draping** | Garment Draper | Wraps the 3D mesh onto the SMPL-X body |
| **3D Render** | PyTorch3D Renderer | RGB render + normal map + depth map |
| **Agnostic Generation** | DWPose + ATR + AgnosticMaskGenerator | Pose + segmentation → garment-stripped person |
| **Garment Encoding** | GarmentNet + IP-Adapter (CLIP Vision) | Features for self-attn fusion and cross-attention |
| **3D Conditioning** | **ControlNet3D** (novel) | 9 channels (RGB+normal+depth) → multi-scale residuals |
| **Backbone** | IDM-VTON (SDXL) | Frozen TryonNet + GarmentNet, with cross-attention |
| **Output** | SDXL VAE Decoder + Post-Processing | Photorealistic try-on result |

---

## 4. Training Strategy — ControlNet3D Only

The project's efficiency key: **the entire backbone is frozen, only ControlNet3D is trained.**

| Component | Parameters | Status |
|---------|-----------|-------|
| TryonNet (SDXL UNet) | ~2.6B | ❄️ Frozen |
| GarmentNet (SDXL UNet) | ~2.6B | ❄️ Frozen |
| VAE (SDXL AutoencoderKL) | ~85M | ❄️ Frozen |
| CLIP Vision + Text Encoders | ~1.8B | ❄️ Frozen |
| IP-Adapter Resampler | ~50M | ❄️ Frozen |
| **ControlNet3D (novel)** | **~350–400M** | ✅ **Trainable** |

Only **~5–7%** of the total parameters are trained. The checkpoint contains only the ControlNet3D
weights (~485MB), under `checkpoints/meshvton/`.

---

## 5. 3D Pipeline (Novel Contribution) — Detail

### 5.1 SMPL-X Body Estimation
The human photo is regressed to SMPL-X parameters:
- `betas (β)` (B,10) — body shape
- `body_pose (θ)` (B,63) — 21 joints × 3 axis-angle
- `global_orient` (B,3), `transl` (B,3)
- `vertices` (B,10475,3), `joints` (B,127,3), `faces` (20908,3)

> The current implementation is a simple ResNet-style regressor (placeholder / `SimpleSMPLXRegressor`).
> The goal is to replace it with PyMAF-X or ExPose in production. **SMPL-X pose estimation from a new
> person is not yet trained** — so back-view / fully pose-consistent draping still needs improvement
> (see Section 11).

### 5.2 Garment Draper
Wraps the 3D garment mesh onto the SMPL-X body.

> **Important fix:** The neural `GarmentDraper` network was untrained and collapsed the mesh into a
> "blob". Instead, **geometric alignment** (`_geometric_align`) was used: convert CLOTH3D's Z-up axis
> to Y-up `(x,y,z) → (x,z,-y)`, center at the origin, scale to the body bounding-box. In addition,
> `.detach()` was required on the `SMPLXEstimator.get_body_mesh` call.

### 5.3 PyTorch3D Differentiable Render
Renders the wrapped mesh from the desired camera angle:
- Cameras: `FoVPerspectiveCameras` (dist=2.7, elev=0, azim ∈ [0,360°])
- Light: `PointLights` + `SoftPhongShader` (ambient + diffuse + specular)
- Outputs: RGB (B,3,H,W), Normal (B,3,H,W), Depth (B,1,H,W → broadcast to 3 channels)

| `azim` | View |
|--------|---------|
| 0° | Front |
| 90° / 270° | Side profile |
| 180° | **Back — what 2D methods cannot do** |

> `render_garment` only renders pairs where the person has SMPL-X **and** the corresponding garment's
> mesh exists; `garment_id` is the name of the parent folder containing the mesh. Pairs are filtered
> to this intersection.

---

## 6. ControlNet3D — The Novel Module

ControlNet3D injects the 3D render outputs (RGB + normal + depth) into TryonNet via multi-scale
residual connections.

### 6.1 Conditioning Input
```
conditioning_3d : (B, 9, H, W)
  ├── RGB render   (B,3,H,W)
  ├── Normal map   (B,3,H,W)
  └── Depth        (B,3,H,W)  (broadcast 1→3 channels)
```

### 6.2 Conditioning Encoder
Downsamples the 9-channel input by 8× to the UNet's base channel width (320):
`Conv(9→16)→…→Conv(256→320, stride=2)` (SiLU at each step). Output: `(B,320,H/8,W/8)`.
Implementation: [src/models/controlnet_3d.py](src/models/controlnet_3d.py).

### 6.3 Encoder Blocks (Mirroring the SDXL UNet)
A ResBlock + Downsample sequence mirroring the SDXL stages; each block output passes through a
`ZeroConv` to produce a residual (12 down residuals + 1 mid residual total). `ControlNet3DResBlock` =
`GroupNorm → SiLU → Conv3×3 → (+timestep_embed) → GroupNorm → SiLU → Conv3×3 → skip`.

### 6.4 Zero-Initialization (Zero-Init)
Each `ZeroConv` is a 1×1 convolution with weight and bias set to **zero**:
- At the start every residual = 0 → the frozen TryonNet is never disturbed (`h + 0 = h`)
- Training gradually moves the weights away from zero
- The standard ControlNet trick (Zhang et al., 2023)

### 6.5 Injection into TryonNet
```python
down_block_additional_residuals = controlnet_residuals[:-1]
mid_block_additional_residual    = controlnet_residuals[-1]
```
Added to the output of each encoder stage and the mid block as `h_i ← h_i + r_i`.

---

## 7. IDM-VTON Backbone (Frozen)

### 7.1 TryonNet — The Main Denoising Backbone
The "hacked" SDXL UNet loaded from `yisol/IDM-VTON` (`src/idm_vton/unet_hacked_tryon.py`).
**13-channel input** (IDM-VTON's signature innovation):
```
unet_in = cat([
    noisy_latent / zt,       # (B,4) noisy person latent
    mask,                    # (B,1) inpaint mask
    masked_image_latents,    # (B,4) agnostic person (VAE)
    pose_img / densepose,    # (B,4) densepose (VAE)
], dim=1)  → (B,13,h,w)
```

> ⚠️ **Critical architectural decision (2026-06):** The channel ordering must be **exactly the same**
> as the `[noise(4), mask(1), masked_image(4), pose/densepose(4)]` layout expected by IDM-VTON's
> frozen backbone. This ordering is defined in the denoising loop in
> [src/idm_vton/tryon_pipeline.py](src/idm_vton/tryon_pipeline.py). (For detail see Section 10.)

### 7.2 GarmentNet — Garment Reference Feature Extractor
A second SDXL UNet (`unet_hacked_garmnet`). It produces `reference_features` from `cloth_lat`; these
are injected into TryonNet via **self-attention fusion** (the `garment_features` parameter).

### 7.3 IP-Adapter — Garment Visual Features
`CLIPVisionModelWithProjection` → penultimate (`hidden_states[-2]`) → Resampler
(dim=1280, depth=4, heads=20, num_queries=16, output=2048) → TryonNet cross-attention.
During training the Resampler lives inside `unet.encoder_hid_proj` and is frozen.

### 7.4 Text Encoders & VAE
- SDXL dual text encoder: `prompt_embeds (B,77,2048)` + pooled `(B,1280)`; used because the SDXL
  format is required even with an empty prompt.
- VAE (SDXL AutoencoderKL), `scaling_factor = 0.13025`. The same VAE encodes three streams: person,
  agnostic, garment/render.

---

## 8. 2D Pre-Processing — Only for the Agnostic Image

The current pipeline does not feed pose maps/DensePose directly to the model (except the densepose
`pose_img` channel). The 2D pre-processing chain exists mainly to produce the
**garment-stripped (agnostic) person image**:

```
Person → PoseEstimator (DWPose) → keypoints (18,3)
       → HumanSegmentation (ATR) → segments (H,W)
       → AgnosticMaskGenerator(image, segments, keypoints) → agnostic image
```

Pre-processing scripts:
- [src/data/preprocessing/extract_pose.py](src/data/preprocessing/extract_pose.py) — DWPose keypoints
- [src/data/preprocessing/extract_segment.py](src/data/preprocessing/extract_segment.py) — ATR body parsing
- [src/data/preprocessing/build_agnostic.py](src/data/preprocessing/build_agnostic.py) — agnostic person
- [src/data/preprocessing/extract_smplx.py](src/data/preprocessing/extract_smplx.py) — SMPL-X parameters + mesh
- [src/data/preprocessing/render_garment.py](src/data/preprocessing/render_garment.py) — RGB + normal + depth

---

## 9. Dataset and Data Flow

| Dataset | Content | Usage |
|-------------|--------|----------|
| [VITON-HD](https://github.com/shadow2496/VITON-HD) | Human photos | Training images |
| [CLOTH3D](https://chalearnlap.cvc.uab.cat/) | 3D garment meshes (OBJ + texture) | 3D garment assets |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/) | Body model parameters | 3D body estimation |

The tensors **`MeshVTONDataset`** ([src/data/dataset.py](src/data/dataset.py)) produces per sample
(all `(height, width)`):

| Key | Shape | Description |
|---------|-------|----------|
| `person` | (3,H,W) [-1,1] | Target: the person wearing the garment |
| `masked_image` | (3,H,W) [-1,1] | Agnostic person |
| `mask` | (1,H,W) [0,1] | Inpaint region (1 = garment area) |
| `pose_img` | (3,H,W) [-1,1] | The person's densepose |
| `cloth` | (3,H,W) [-1,1] | 3D garment rendered to the person's pose |
| `conditioning_3d` | (9,H,W) [-1,1] | render(3)+normal(3)+depth(3) → ControlNet3D |

Expected directory layout:
```
{data_root}/raw/images/{person_id}.jpg
{data_root}/processed/agnostic/{person_id}.jpg
{data_root}/processed/densepose/{person_id}.jpg|png
{data_root}/processed/renders_3d/{person_id}_{garment_id}.png
{data_root}/processed/normal_maps/{person_id}_{garment_id}.png
{data_root}/processed/depth_maps/{person_id}_{garment_id}.png
```

> `renders_3d` (the scaled/rendered garment) serves as both the GarmentNet input (`cloth`) and the
> ControlNet3D conditioning.

---

## 10. Training — On the Real IDM-VTON Pipeline (Phase 2)

Training is done with [scripts/train_meshvton.py](scripts/train_meshvton.py) on the **frozen real
IDM-VTON pipeline**, using IDM-VTON's full forward contract. Per step (no CFG):

```
z0          = vae(person) · scaling                      # target latent
zt          = scheduler.add_noise(z0, eps, t)
unet_in     = cat([zt, mask, vae(masked_image), vae(pose_img)])   # 13 channels
ref_feats   = GarmentNet(vae(cloth), t, text_c)                   # self-attn reference
image_embeds= unet.encoder_hid_proj(image_encoder(clip(cloth))[-2])  # IP-Adapter
residuals   = ControlNet3D(conditioning_3d, t)            # 12 down + 1 mid (TRAINABLE)
eps_pred    = unet(unet_in, t, text, added_cond_kwargs,
                   down/mid residuals, garment_features=ref_feats)
loss        = MSE(eps_pred, eps)
```

Only `ControlNet3D.parameters()` receive gradients. Run:
```bash
python scripts/train_meshvton.py --data_root data --pairs data/raw/train_pairs.csv
```

### Hyperparameters (summary)
| Parameter | Value |
|-----------|-------|
| Resolution | 512 × 384 (train script) / 512 × 512 (config) |
| Latent | 4 × 64 × 64 |
| Optimizer | AdamW (lr=1e-4 script / 1e-5 config), weight_decay=0.01 |
| LR schedule | cosine + 500-step warmup |
| Precision | bf16 (script) / fp16 (config) |
| Grad accumulation | 16 (effective batch 16) |
| Max grad norm | 1.0 |
| Diffusion steps | 1000 (train, DDPM) / 50 (infer, DDIM) |
| EMA decay | 0.9999 (ControlNet3D only) |

> The loss is currently pure **MSE(ε̂, ε)**. Additional VGG perceptual / LPIPS / adversarial / KL
> terms are defined inside `TryOnLoss`, but the default forward returns only the noise MSE.

---

## 11. ⚠️ Important Architectural Decision: The Custom Pipeline Was Abandoned

> This is a **hidden but critical** fact in the codebase, and some parts of `docs/ARCHITECTURE.md`
> still refer to the old custom pipeline.

- The hand-written [src/models/tryon_pipeline.py](src/models/tryon_pipeline.py) (custom
  `TryOnPipeline.forward`/`generate`) is **BROKEN and abandoned for producing results.**
  - The custom 13-channel layout `[noise, agnostic, mask, garment]` did not match the
    `[noise(4), mask(1), masked_image(4), pose/densepose(4)]` layout expected by IDM-VTON's frozen
    backbone → the frozen `conv_in` got confused → pure noise output.
- **The working approach:** the real `StableDiffusionXLInpaintPipeline` in
  [src/idm_vton/tryon_pipeline.py](src/idm_vton/tryon_pipeline.py) (verified: produces clean try-on).
- **Status (2026-06-17):** All stages verified end to end; the first clean MeshVTON result obtained —
  the shape of a 3D bermuda-shorts mesh transferred to the person, with no noise.

**Remaining work / known limitations:**
- Rendering the whole dataset + longer training for quality.
- **SMPL-X pose estimation from a new person is untrained** (`SimpleSMPLXRegressor`) → fully
  pose-consistent / back-view draping still needs improvement.

---

## 12. Inference

### 12.1 3D-Aware Mode (Novel)
```
1. Pre-processing: agnostic image
2. 3D pipeline: SMPL-X → load mesh → drape → render(view_angle)
   conditioning_3d = cat(rgb, normal, depth)  # (B,9,H,W)
3. The garment's front render is used as garment_image (GarmentNet/IP-Adapter sees the correct geometry)
4. DDIM loop (50 steps) + ControlNet3D residuals at each step
5. VAE decode + post-process
```

### 12.2 Multi-Angle Generation
```bash
python scripts/inference.py --person p.jpg --garment g.obj --view_angle 0    # front
python scripts/inference.py --person p.jpg --garment g.obj --view_angle 90   # side
python scripts/inference.py --person p.jpg --garment g.obj --view_angle 180  # back (2D cannot)
```

### 12.3 Classifier-Free Guidance
`ε̂ = ε_uncond + w·(ε_cond − ε_uncond)`, `w = 7.5`. Post-processing: face restoration (CodeFormer),
edge smoothing, color correction, optional SynthID watermark.

---

## 13. Runtime Environment (Colab)

- **Hardware:** RTX PRO 6000 Blackwell, 102 GB. Python 3.12, torch 2.11.0+cu128, CUDA 12.8, nvcc available.
- **Pinned versions (IDM-VTON compatibility):**
  `diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0 huggingface_hub==0.20.3 peft==0.7.1`
  - `huggingface_hub 0.20.3`: newer versions remove `cached_download`, which diffusers 0.25 imports.
  - `peft 0.7.1`: newer versions import `clear_device_cache`, which is not in accelerate 0.25.
- **detectron2 (for densepose):** no prebuilt wheel for torch 2.11 → build from source:
  `FORCE_CUDA=1 pip install 'git+https://github.com/facebookresearch/detectron2.git'` (compiles and
  runs; downgrades iopath to 0.1.9).
- **Pre-processing (densepose + mask):** the `yisol/IDM-VTON` GitHub repo is cloned (`apply_net.py`,
  `preprocess/`, `utils_mask.get_mask_location`). LFS pull fails due to quota → the model files are
  downloaded directly (densepose `model_final_162be9.pkl`, humanparsing `.onnx`, openpose
  `body_pose_model.pth`). Also `pip install av onnxruntime-gpu`.
- **`src` name collision:** Both MeshVTON and IDM-VTON-official have a `src/` package. The IDM-VTON
  pipeline is imported via our own copy `src.idm_vton.*`, with `/content/MeshVTON` first on sys.path
  (after first clearing any cached `src` from sys.modules).
- **Notebooks:** Since the inference notebook imports modules into the kernel, a code change after
  `git pull` requires a kernel restart / module reload (train.py runs as a subprocess, so it is not
  affected).

---

## 14. Project Structure

```
MeshVTON/
├── configs/
│   ├── train.yaml / inference.yaml
│   └── data/              # dataset.yaml, preprocessing.yaml
├── src/
│   ├── idm_vton/          # ✅ REAL working IDM-VTON pipeline (hacked UNets)
│   │   ├── tryon_pipeline.py       # StableDiffusionXLInpaintPipeline (working)
│   │   ├── unet_hacked_tryon.py    # TryonNet (13-ch, residual + garment_features)
│   │   └── unet_hacked_garmnet.py  # GarmentNet
│   ├── models/            # controlnet_3d.py (novel) + tryon_pipeline.py (BROKEN/abandoned)
│   ├── modules/           # smplx_estimator, garment_draper, mesh_renderer,
│   │                      # pose_estimator, segmentation, agnostic_mask, warping, garment_encoder
│   ├── training/          # trainer, losses, lr_scheduler, ema
│   ├── inference/         # image_tryon, video_tryon, postprocess
│   ├── video/             # motion_module, temporal_attention, frame_interpolation, physics_prior
│   └── data/              # dataset, transforms, pair_sampler, preprocessing/
├── scripts/               # train_meshvton.py (✅), train.py, inference.py, evaluate.py,
│                          # preprocess_dataset.py, setup_data.py, export_onnx.py, zip_for_drive.py
├── ip_adapter/            # attention_processor, resampler, ip_adapter, utils
├── notebooks/             # MeshVTON_Train.ipynb, meshvton_inference.ipynb
├── docs/                  # ARCHITECTURE.md (partly outdated), DATASET_GUIDE.md, diagram
├── tests/                 # test_pipeline.py, test_warping.py
├── data/                  # datasets (gitignored)
└── checkpoints/           # weights (gitignored); meshvton/ → ControlNet3D-only ~485MB
```

> **Note:** The distinction between `src/idm_vton/` (working) and `src/models/tryon_pipeline.py`
> (abandoned) is critical — new development must be done through `src/idm_vton/`.

---

## 15. Key Hyperparameter Summary

| Parameter | Value | Note |
|-----------|-------|-----|
| Image resolution | 512×512 / 512×384 | config / train script |
| Latent | 4×64×64 | SDXL VAE |
| VAE scaling | 0.13025 | SDXL (SD1.5: 0.18215) |
| UNet channel multipliers | (1,2,4,4) | 320→640→1280→1280 |
| ControlNet3D input | 9 channels | RGB(3)+Normal(3)+Depth(3) |
| TryonNet input | **13 channels** | noise(4)+mask(1)+masked(4)+pose(4) |
| IP-Adapter tokens | 16 | dim=2048 |
| Diffusion steps | 1000 train / 50 infer | DDPM / DDIM |
| Guidance | 7.5 | CFG |
| Trainable params | ~350–400M | ~5–7% |

---

## 16. Current Status & Roadmap

**✅ Completed (as of 2026-06-17):**
- Phase 1: ControlNet3D residual injection inside the pipeline `__call__` (None-safe, the 2D path is
  not broken).
- Phase 2: ControlNet3D training on the frozen real pipeline via `MeshVTONDataset` +
  `train_meshvton.py` (IDM-VTON's full 13-ch forward, trained Resampler, GarmentNet reference
  features, noise MSE).
- Geometric draping/render fixes (blob problem solved).
- First clean end-to-end MeshVTON result.

**🔜 Remaining / Goals:**
- Render the whole dataset + longer training for quality.
- Integrate a real SMPL-X pose estimation module (PyMAF-X / ExPose) → pose-/back-accurate draping.
- Enable the auxiliary loss terms (LPIPS, perceptual).
- Clean `docs/ARCHITECTURE.md` of custom-pipeline references and update it to the working
  `src/idm_vton/` path.
- The video try-on path (`src/video/`, `src/inference/video_tryon.py`) is still experimental.

---

## 17. License & Citation

Apache 2.0.
```bibtex
@software{MeshVTON2025,
  title={MeshVTON: Geometry-Aware Virtual Try-On},
  author={Serhan Telatar},
  year={2025},
  url={https://github.com/SerhanTelatar/MeshVTON}
}
```
