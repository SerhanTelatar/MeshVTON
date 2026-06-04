"""
Data Setup Script — Reorganize downloaded datasets for MeshVTON pipeline.

Does the following:
1. Moves VITON-HD person images and preprocessing outputs to pipeline structure
2. Organizes CLOTH3D garment meshes by category
3. Renames SMPL-X model file to expected name
4. Creates train_pairs.csv matching persons to garments
"""

import shutil
import csv
import random
from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Auto-detect VITON-HD location (may be nested under archive/)
_archive = PROJECT_ROOT / "data" / "raw" / "images" / "archive"
if (_archive / "train").exists():
    VITONHD_ROOT = _archive
elif (_archive / "image").exists():
    VITONHD_ROOT = _archive
else:
    VITONHD_ROOT = _archive

CLOTH3D_ROOT = PROJECT_ROOT / "data" / "garments_3d" / "val_t1" / "val_t1"
SMPLX_DIR = PROJECT_ROOT / "checkpoints" / "pretrained" / "smplx"

# Target directories
DATA_ROOT = PROJECT_ROOT / "data"
RAW_IMAGES = DATA_ROOT / "raw" / "images"
PROCESSED = DATA_ROOT / "processed"
GARMENTS_3D = DATA_ROOT / "garments_3d"

# Garment category mapping
GARMENT_CATEGORY = {
    "Top": "upper_body",
    "Tshirt": "upper_body",
    "Trousers": "lower_body",
    "Skirt": "lower_body",
    "Dress": "dresses",
    "Jumpsuit": "outerwear",
}


def setup_directories():
    """Create all necessary directories."""
    dirs = [
        RAW_IMAGES,
        PROCESSED / "poses",
        PROCESSED / "poses" / "rendered",
        PROCESSED / "segments",
        PROCESSED / "agnostic",
        PROCESSED / "smplx_params",
        PROCESSED / "smplx_meshes",
        PROCESSED / "renders_3d",
        PROCESSED / "normal_maps",
        PROCESSED / "depth_maps",
        GARMENTS_3D / "upper_body",
        GARMENTS_3D / "lower_body",
        GARMENTS_3D / "dresses",
        GARMENTS_3D / "outerwear",
        DATA_ROOT / "raw",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("Directory structure created")


def reorganize_vitonhd():
    """Move VITON-HD files to pipeline-expected locations."""
    train_dir = VITONHD_ROOT / "train"
    test_dir = VITONHD_ROOT / "test"

    if not train_dir.exists():
        print("  VITON-HD train folder not found, skipping")
        return 0

    count = 0

    # --- Person images → data/raw/images/ ---
    src_images = train_dir / "image"
    if src_images.exists():
        for img in src_images.iterdir():
            if img.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                dest = RAW_IMAGES / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
                    count += 1
        print(f"  Person images copied: {count} files")

    # --- OpenPose → data/processed/poses/ ---
    src_pose_img = train_dir / "openpose_img"
    src_pose_json = train_dir / "openpose_json"
    if src_pose_img and src_pose_img.exists():
        pose_count = 0
        for f in src_pose_img.iterdir():
            dest = PROCESSED / "poses" / "rendered" / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                pose_count += 1
        print(f"  Pose renders copied: {pose_count} files")
    if src_pose_json and src_pose_json.exists():
        json_count = 0
        for f in src_pose_json.iterdir():
            dest = PROCESSED / "poses" / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                json_count += 1
        print(f"  Pose JSONs copied: {json_count} files")

    # --- Segmentation → data/processed/segments/ ---
    src_seg = train_dir / "image-parse-v3"
    if src_seg and src_seg.exists():
        seg_count = 0
        for f in src_seg.iterdir():
            dest = PROCESSED / "segments" / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                seg_count += 1
        print(f"  Segmentation maps copied: {seg_count} files")

    # --- Agnostic → data/processed/agnostic/ ---
    src_agn = train_dir / "agnostic-v3.2"
    if src_agn and src_agn.exists():
        agn_count = 0
        for f in src_agn.iterdir():
            dest = PROCESSED / "agnostic" / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                agn_count += 1
        print(f"  Agnostic masks copied: {agn_count} files")

    # Also process test set
    if test_dir and test_dir.exists():
        test_img = test_dir / "image"
        if test_img.exists():
            test_count = 0
            for img in test_img.iterdir():
                if img.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                    dest = RAW_IMAGES / img.name
                    if not dest.exists():
                        shutil.copy2(img, dest)
                        test_count += 1
            print(f"  Test person images copied: {test_count} files")

    return count


def reorganize_cloth3d():
    """Organize CLOTH3D garment meshes by category."""
    if not CLOTH3D_ROOT.exists():
        print("  CLOTH3D folder not found, skipping")
        return {}

    garment_index = {}  # garment_id → {"category": ..., "type": ..., "path": ...}
    total = 0

    for seq_dir in sorted(CLOTH3D_ROOT.iterdir()):
        if not seq_dir.is_dir():
            continue

        seq_id = seq_dir.name

        # Find all OBJ files in this sequence
        for obj_file in seq_dir.glob("*.obj"):
            garment_type = obj_file.stem  # "Dress", "Top", etc.
            category = GARMENT_CATEGORY.get(garment_type, "upper_body")

            garment_id = f"{seq_id}_{garment_type}"
            dest_dir = GARMENTS_3D / category / garment_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy OBJ mesh
            dest_obj = dest_dir / "mesh.obj"
            if not dest_obj.exists():
                shutil.copy2(obj_file, dest_obj)

            # Copy texture PNG if it exists
            png_file = obj_file.with_suffix(".png")
            if png_file.exists():
                dest_tex = dest_dir / "texture.png"
                if not dest_tex.exists():
                    shutil.copy2(png_file, dest_tex)

            # Copy info.mat if it exists
            info_file = seq_dir / "info.mat"
            if info_file.exists():
                dest_info = dest_dir / "info.mat"
                if not dest_info.exists():
                    shutil.copy2(info_file, dest_info)

            garment_index[garment_id] = {
                "category": category,
                "type": garment_type,
                "path": str(dest_dir),
            }
            total += 1

    print(f"  CLOTH3D garments organized: {total} items")

    # Print category breakdown
    categories = {}
    for info in garment_index.values():
        cat = info["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"     {cat}: {count} items")

    return garment_index


def fix_smplx_filename():
    """Rename SMPLX_NEUTRAL_2020.npz to match what smplx library expects."""
    src = SMPLX_DIR / "SMPLX_NEUTRAL_2020.npz"
    # smplx library looks for SMPLX_NEUTRAL.npz
    dest = SMPLX_DIR / "SMPLX_NEUTRAL.npz"

    if src.exists() and not dest.exists():
        shutil.copy2(src, dest)
        print("  Copied SMPLX_NEUTRAL_2020.npz → SMPLX_NEUTRAL.npz")
    elif dest.exists():
        print("  SMPLX_NEUTRAL.npz already present")
    else:
        print("  SMPL-X model file not found!")


def create_pairs_csv(garment_index: dict):
    """Create train/val/test pairs CSV files."""
    # Get list of person images
    person_ids = sorted([
        f.stem for f in RAW_IMAGES.iterdir()
        if f.suffix.lower() in {".jpg", ".png", ".jpeg"}
        and not f.name.startswith(".")
    ])

    if not person_ids:
        print("  No person images found, cannot create CSVs")
        return

    garment_ids = sorted(garment_index.keys())

    if not garment_ids:
        print("  No garment meshes found, cannot create CSVs")
        return

    # Create pairs: each person with multiple random garments
    pairs = []
    for person_id in person_ids:
        # Assign 3-5 random garments per person
        num_garments = min(random.randint(3, 5), len(garment_ids))
        selected = random.sample(garment_ids, num_garments)
        for garment_id in selected:
            pairs.append((person_id, garment_id))

    random.shuffle(pairs)

    # Split: 90% train, 5% val, 5% test
    n = len(pairs)
    n_train = int(n * 0.9)
    n_val = int(n * 0.05)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    # Write CSVs
    raw_dir = DATA_ROOT / "raw"
    for name, data in [("train_pairs.csv", train_pairs),
                        ("val_pairs.csv", val_pairs),
                        ("test_pairs.csv", test_pairs)]:
        path = raw_dir / name
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["person_id", "garment_id"])
            writer.writerows(data)

    print(f"  Pair files created:")
    print(f"     train: {len(train_pairs)} pairs")
    print(f"     val:   {len(val_pairs)} pairs")
    print(f"     test:  {len(test_pairs)} pairs")
    print(f"     Total persons: {len(person_ids)}, total garments: {len(garment_ids)}")


def main():
    print("=" * 60)
    print("  MeshVTON Data Organization")
    print("=" * 60)
    print()

    print("1. Creating directory structure...")
    setup_directories()
    print()

    print("2. Organizing VITON-HD files...")
    reorganize_vitonhd()
    print()

    print("3. Organizing CLOTH3D garment meshes...")
    garment_index = reorganize_cloth3d()
    print()

    print("4. Fixing SMPL-X model filename...")
    fix_smplx_filename()
    print()

    print("5. Creating training pairs (train_pairs.csv)...")
    create_pairs_csv(garment_index)
    print()

    print("=" * 60)
    print("  Data organization complete")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. pip install smplx trimesh pytorch3d")
    print("  2. python src/data/preprocessing/extract_smplx.py")
    print("  3. python src/data/preprocessing/render_garment.py")


if __name__ == "__main__":
    main()
