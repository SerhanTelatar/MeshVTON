"""Zip data files into drive/ folder for Google Drive upload."""
import zipfile
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DRIVE = PROJECT / "drive"
DRIVE.mkdir(exist_ok=True)


def zip_folder(src_dir, zip_name, extensions=None):
    """Zip a folder's contents."""
    src = Path(src_dir)
    if not src.exists():
        print(f"  ⚠️ {src_dir} bulunamadı, atlıyorum")
        return
    
    out = DRIVE / zip_name
    if out.exists():
        print(f"  ⏭️ {zip_name} zaten mevcut, atlıyorum")
        return
    
    print(f"  📦 {zip_name} oluşturuluyor...")
    count = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as zf:  # ZIP_STORED = sıkıştırma yok (hızlı)
        for root, dirs, files in os.walk(src):
            for f in files:
                fp = Path(root) / f
                if extensions and fp.suffix.lower() not in extensions:
                    continue
                arcname = fp.relative_to(src.parent)
                zf.write(fp, arcname)
                count += 1
                if count % 1000 == 0:
                    print(f"    ... {count} dosya eklendi")
    
    size_mb = out.stat().st_size / 1e6
    print(f"  ✅ {zip_name} → {count} dosya, {size_mb:.0f} MB")


def zip_images(src_dir, zip_name):
    """Zip only image files (fast, no compression)."""
    src = Path(src_dir)
    out = DRIVE / zip_name
    if out.exists():
        print(f"  ⏭️ {zip_name} zaten mevcut, atlıyorum")
        return
    
    print(f"  📦 {zip_name} oluşturuluyor...")
    count = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as zf:
        for f in sorted(src.iterdir()):
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                zf.write(f, f"images/{f.name}")
                count += 1
                if count % 2000 == 0:
                    print(f"    ... {count} dosya eklendi")
    
    size_mb = out.stat().st_size / 1e6
    print(f"  ✅ {zip_name} → {count} dosya, {size_mb:.0f} MB")


if __name__ == "__main__":
    print("=" * 50)
    print("  Drive Upload ZIP'leri Oluşturuluyor")
    print("=" * 50)
    print()
    
    # 1. Kişi görüntüleri
    zip_images(PROJECT / "data/raw/images", "images.zip")
    
    # 2. Preprocessing çıktıları
    zip_folder(PROJECT / "data/processed/poses", "poses.zip")
    zip_folder(PROJECT / "data/processed/segments", "segments.zip")
    zip_folder(PROJECT / "data/processed/densepose", "densepose.zip")
    zip_folder(PROJECT / "data/processed/agnostic", "agnostic.zip")
    
    # 3. 3D garment meshes
    zip_folder(PROJECT / "data/garments_3d", "garments_3d.zip", 
               extensions={'.obj', '.png', '.mat', '.json'})
    
    # 4. SMPL-X + VPoser
    zip_folder(PROJECT / "checkpoints/pretrained", "pretrained.zip")
    
    print()
    print("=" * 50)
    print("  drive/ klasöründeki dosyalar:")
    print("=" * 50)
    total = 0
    for f in sorted(DRIVE.iterdir()):
        size = f.stat().st_size / 1e6
        total += size
        print(f"  {f.name:30s} {size:>8.1f} MB")
    print(f"  {'TOPLAM':30s} {total:>8.1f} MB")
