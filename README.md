# DiffFit-3D

**Multi-View Virtual Try-On via 3D Garment Mesh Conditioning in Latent Diffusion Models**

DiffFit-3D is a virtual try-on pipeline that combines 3D human body estimation (SMPL-X), differentiable garment rendering (PyTorch3D), and a pre-trained IDM-VTON backbone (SDXL-based Latent Diffusion) to achieve geometrically accurate, photorealistic garment fitting across diverse body shapes and camera perspectives.

## Architecture

![DiffFit-3D Architecture](docs/architecture_diagram.png)

### Pipeline Overview

| Stage | Component | Description |
|-------|-----------|-------------|
| **Input** | Person Image + 3D Garment Mesh | 2D photo of person + `.obj` garment file |
| **Body Estimation** | SMPL-X Estimator | Extracts 3D body shape, pose, and joints |
| **Garment Fitting** | Garment Draper | Drapes 3D garment mesh onto SMPL-X body |
| **3D Rendering** | PyTorch3D Renderer | Generates RGB render, normal map, depth map |
| **3D Conditioning** | **ControlNet3D** (novel) | Injects 3D cues into the diffusion backbone |
| **2D Conditioning** | DWPose + DensePose + ATR | Pose keypoints, UV maps, segmentation |
| **Backbone** | IDM-VTON (SDXL) | Frozen TryonNet + GarmentNet with cross-attention |
| **Output** | SDXL VAE Decoder | Photorealistic try-on result |

### What Makes It Different

Unlike 2D-only virtual try-on methods, DiffFit-3D uses **real 3D garment meshes** instead of 2D garment photos. This means:

- 🔄 **Back & side views** — The model renders the garment from any angle, no hallucination needed
- 📐 **Geometric accuracy** — 3D mesh ensures correct proportions across all body types
- 🎭 **Normal & depth maps** — Provide the diffusion model with 3D structural cues via ControlNet3D
- 💡 **Physically-based lighting** — Phong shading produces realistic shadows and highlights
- 🧠 **IDM-VTON backbone** — Pre-trained SDXL-based try-on model provides state-of-the-art generation quality

### Training Strategy

| Component | Parameters | Status |
|-----------|-----------|--------|
| **TryonNet** (SDXL UNet) | ~2.6B | **Frozen** |
| **GarmentNet** (SDXL UNet) | ~2.6B | **Frozen** |
| **VAE** (SDXL) | ~85M | **Frozen** |
| **ControlNet3D** (novel) | ~350-400M | **Trainable** ✅ |

Only ~6-7% of total parameters are trained — the 3D ControlNet conditioning module.

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/SerhanTelatar/DiffFit-3D.git
cd DiffFit-3D

# Core dependencies
pip install -r requirements.txt

# 3D Pipeline dependencies
pip install smplx trimesh
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Dataset Setup

1. **Download datasets** (see [Dataset Guide](docs/DATASET_GUIDE.md)):
   - Person images: [VITON-HD](https://github.com/shadow2496/VITON-HD) → `data/raw/images/`
   - 3D garment meshes: [CLOTH3D](https://chalearnlap.cvc.uab.cat/) → `data/garments_3d/`
   - SMPL-X model: [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) → `checkpoints/pretrained/smplx/`

2. **Organize data**:
```bash
python scripts/setup_data.py
```

3. **Preprocess** (requires GPU):
```bash
python src/data/preprocessing/extract_smplx.py
python src/data/preprocessing/render_garment.py
```

### Training

```bash
# Train with IDM-VTON backbone (downloads from HuggingFace)
python scripts/train.py --config configs/train.yaml

# Local dev mode (no download, placeholder models)
python scripts/train.py --config configs/train.yaml --local
```

For Colab training, see [`notebooks/DiffFit3D_Train.ipynb`](notebooks/DiffFit3D_Train.ipynb).

### Inference

```bash
# 3D garment mesh (novel — renders from any angle)
python scripts/inference.py \
    --person path/to/person.jpg \
    --garment path/to/garment.obj \
    --output results/output.png \
    --view_angle 0

# Back view
python scripts/inference.py \
    --person path/to/person.jpg \
    --garment path/to/garment.obj \
    --output results/back_view.png \
    --view_angle 180

# Traditional 2D garment image
python scripts/inference.py \
    --person path/to/person.jpg \
    --garment path/to/garment.jpg \
    --output results/output.png
```

## Project Structure

```
DiffFit-3D/
├── configs/
│   ├── data/              # Dataset & preprocessing configs
│   └── model/             # UNet, attention, pipeline configs
├── src/
│   ├── models/            # IDM-VTON backbone + ControlNet3D
│   │   ├── tryon_pipeline.py  # Main pipeline (IDM-VTON + ControlNet3D)
│   │   ├── controlnet_3d.py   # 3D conditioning module (NOVEL)
│   │   └── attention/         # Self, cross, spatial attention
│   ├── modules/           # SMPL-X estimator, mesh renderer,
│   │                      # garment draper, pose, segmentation
│   ├── training/          # Training loop, losses, schedulers
│   ├── inference/         # Image try-on (2D + 3D modes)
│   └── data/              # Dataset, transforms, preprocessing
│       └── preprocessing/ # SMPL-X extraction, 3D rendering
├── scripts/               # CLI: train, inference, data setup
├── notebooks/             # Colab training notebook
├── docs/                  # Architecture docs & guides
├── data/                  # Datasets (gitignored)
│   ├── raw/images/        # Person photos (VITON-HD)
│   ├── garments_3d/       # 3D garment meshes (CLOTH3D)
│   └── processed/         # Preprocessing outputs
└── checkpoints/           # Model weights (gitignored)
    └── pretrained/
        ├── smplx/         # SMPLX_NEUTRAL.npz
        └── vposer/        # VPoser v1.0
```

## Datasets

| Dataset | Content | Size | Usage |
|---------|---------|------|-------|
| [VITON-HD](https://github.com/shadow2496/VITON-HD) | Person photos + preprocessing | ~12 GB | Training images |
| [CLOTH3D](https://chalearnlap.cvc.uab.cat/) | 3D garment meshes (OBJ + textures) | ~12 GB (val) | 3D garment assets |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/) | Body model parameters | ~170 MB | 3D body estimation |

## Technical Details

### 3D Pipeline (Novel Contribution)

1. **SMPL-X Body Estimation** → Predicts body shape (β), pose (θ), and expression from person photo
2. **Garment Draping** → Coarse alignment + neural fine-tuning + collision handling
3. **Differentiable Rendering** → PyTorch3D Phong renderer produces RGB + normal + depth maps
4. **ControlNet3D Conditioning** → 9-channel (RGB+normal+depth) → multi-scale residuals → TryonNet injection

### IDM-VTON Backbone

- **TryonNet**: SDXL UNet (2.6B params, frozen) — main denoising backbone
- **GarmentNet**: SDXL UNet (2.6B params, frozen) — garment feature extraction
- **Cross-Attention Fusion**: Garment features injected via IP-Adapter + cross-attention
- **Scheduler**: DDPM training / DDIM inference (50 steps)
- **Guidance**: Classifier-free guidance (w=7.5)

## Citation

```bibtex
@software{difffit3d2025,
    title={DiffFit-3D: Geometry-Aware Virtual Try-On},
    author={Serhan Telatar},
    year={2025},
    url={https://github.com/SerhanTelatar/DiffFit-3D}
}
```

## License

Apache 2.0
