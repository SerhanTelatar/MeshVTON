# MeshVTON: Dataset Preparation Guide

## 1. Data to Download

The full MeshVTON 3D pipeline requires **three types of data**:

---

### 1.1 — Person Images (2D)

Person images from any virtual try-on dataset:

| Dataset | Download | Size | Note |
|---------|----------|------|------|
| **VITON-HD** | [GitHub](https://github.com/shadow2496/VITON-HD) | ~12 GB | Most common benchmark |
| **DressCode** | [GitHub](https://github.com/aimagelab/dress-code) | ~50 GB | Upper / Lower / Dress |
| **DeepFashion** | [mmlab.ie.cuhk.edu.hk](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) | ~30 GB | Large-scale |

**Steps**: Go to the dataset's web page → fill the academic-use form → obtain the download link.

---

### 1.2 — 3D Garment Meshes

| Source | Download | Format | Note |
|--------|----------|--------|------|
| **CLOTH3D** | [cloth3d.github.io](https://chalearnlap.cvc.uab.cat/dataset/38/description/) | OBJ | Synthetic, SMPL-compatible |
| **Deep Fashion3D** | [GitHub](https://github.com/kv2000/DeepFashion3D) | OBJ/PLY | Real scanned |
| **ClothesNet** | [clothesnet.github.io](https://clothesnet.github.io/) | OBJ | 3000+ garments |
| **Sketchfab** | [sketchfab.com](https://sketchfab.com/search?q=clothing&type=models) | GLB/OBJ | Free models available |
| **TurboSquid** | [turbosquid.com](https://www.turbosquid.com/Search/3D-Models/free/clothing) | OBJ/FBX | Free / paid |
| **CGTrader** | [cgtrader.com](https://www.cgtrader.com/free-3d-models/clothes) | OBJ | Free models |

**Best starting point**: **CLOTH3D** — synthetic, labeled, and compatible with the SMPL body model.

---

### 1.3 — SMPL-X Body Model

| File | Download | Note |
|------|----------|------|
| **SMPL-X model** | [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) | Registration required (academic) |
| Required files | `SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz` | ~300 MB |

**Steps**: smpl-x.is.tue.mpg.de → register → Download SMPL-X → grab the `SMPLX_*.npz` files.

---

## 2. File Layout

Place the downloaded files following this structure:

```
MeshVTON/
├── checkpoints/
│   └── pretrained/
│       └── smplx/                          ← SMPL-X MODEL FILES
│           ├── SMPLX_NEUTRAL.npz           ← Download from smpl-x.is.tue.mpg.de
│           ├── SMPLX_MALE.npz
│           └── SMPLX_FEMALE.npz
│
├── data/
│   ├── raw/
│   │   ├── images/                         ← PERSON IMAGES (2D)
│   │   │   ├── 00001_00.jpg                   from VITON-HD
│   │   │   ├── 00002_00.jpg
│   │   │   └── ...
│   │   ├── train_pairs.csv                 ← TRAINING PAIRS
│   │   ├── val_pairs.csv
│   │   └── test_pairs.csv
│   │
│   ├── garments_3d/                        ← 3D GARMENT MESHES
│   │   ├── upper_body/
│   │   │   ├── tshirt_001/
│   │   │   │   ├── mesh.obj                   3D geometry
│   │   │   │   ├── texture.png                UV texture map
│   │   │   │   └── metadata.json              Category and material info
│   │   │   ├── shirt_002/
│   │   │   │   ├── mesh.obj
│   │   │   │   └── texture.png
│   │   │   └── ...
│   │   ├── lower_body/
│   │   │   ├── pants_001/
│   │   │   │   ├── mesh.obj
│   │   │   │   └── texture.png
│   │   │   └── ...
│   │   ├── dresses/
│   │   │   └── ...
│   │   └── outerwear/
│   │       └── ...
│   │
│   └── processed/                          ← AUTO-GENERATED
│       ├── poses/                             pose keypoints
│       ├── segments/                          body segmentation
│       ├── agnostic/                          agnostic masks
│       ├── smplx_params/                      SMPL-X body parameters
│       ├── smplx_meshes/                      SMPL-X body meshes (.obj)
│       ├── renders_3d/                        rendered garment images
│       ├── normal_maps/                       normal maps
│       └── depth_maps/                        depth maps
```

---

## 3. CSV File Format

The `train_pairs.csv` file pairs persons with 3D garment meshes:

```csv
person_id,garment_id
00001_00,tshirt_001
00002_00,shirt_002
00003_00,tshirt_001
00001_00,dress_001
...
```

- **person_id**: filename (without extension) under `data/raw/images/`
- **garment_id**: folder name under `data/garments_3d/*/`

---

## 4. Garment Metadata Format

Each garment folder contains a `metadata.json`:

```json
{
    "name": "Basic White T-Shirt",
    "category": "upper_body",
    "subcategory": "tshirt",
    "material": {
        "type": "cotton",
        "weight": 0.3,
        "stiffness": 0.4,
        "stretch": 0.6,
        "friction": 0.5
    },
    "mesh": {
        "format": "obj",
        "vertices": 5234,
        "faces": 10420,
        "has_uv": true,
        "has_texture": true,
        "scale": "meters"
    },
    "tags": ["casual", "unisex", "short_sleeve"]
}
```

---

## 5. Preprocessing Commands

Once the downloads are in place, run the full preprocessing chain:

```bash
# 1. 2D preprocessing — pose, segmentation, agnostic mask
python scripts/preprocess_dataset.py --steps pose segment agnostic

# 2. Extract SMPL-X body parameters (3D)
python src/data/preprocessing/extract_smplx.py \
    --image_dir data/raw/images \
    --output_dir data/processed/smplx_params \
    --model_dir checkpoints/pretrained/smplx \
    --mesh_dir data/processed/smplx_meshes \
    --save_mesh

# 3. Render 3D garments (3D)
python src/data/preprocessing/render_garment.py \
    --garments_dir data/garments_3d \
    --smplx_params_dir data/processed/smplx_params \
    --output_dir data/processed/renders_3d \
    --normal_maps_dir data/processed/normal_maps \
    --depth_maps_dir data/processed/depth_maps \
    --resolution 512
```

---

## 6. Required Libraries (3D Dependencies)

```bash
# Extra packages for the 3D pipeline
pip install smplx trimesh pytorch3d

# PyTorch3D may require CUDA compilation:
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

---

## 7. Minimum Starter Dataset

Minimum requirements for a quick test:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Person images | 10 | 1000+ |
| 3D garment meshes | 3 | 50+ |
| SMPL-X model | 1 (neutral) | 3 (neutral + male + female) |

---

## FAQ

**Q: My 3D mesh has no UV map — is that a problem?**
A: No, the system falls back to vertex coloring automatically. UV textures produce better results though.

**Q: Can I create my own 3D models?**
A: Yes! You can build garment models in Blender, CLO3D, or Marvelous Designer and export to OBJ.

**Q: Will it work without SMPL-X?**
A: The system uses a placeholder mesh, but the results will be poor. Downloading SMPL-X is strongly recommended.
