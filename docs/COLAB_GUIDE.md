# DiffFit-3D: Colab'da Eğitim — Adım Adım Rehber

## 📋 Genel Plan

```
                  LOKAL (Senin PC)              →  GOOGLE DRIVE  →  COLAB (GPU)
┌──────────────────────────────────────┐                           ┌────────────────────┐
│ data/raw/images/ (13K fotoğraf)      │    ZIP + Upload           │ Preprocessing      │
│ data/garments_3d/ (818 mesh)         │  ──────────────►          │ SMPL-X extraction  │
│ data/processed/ (pose,seg,dp,agn)    │                           │ 3D garment render  │
│ checkpoints/pretrained/smplx/        │                           │ Eğitim             │
│ src/ (model kodu)                    │                           │ Sonuçları kaydet   │
└──────────────────────────────────────┘                           └────────────────────┘
```

---

## Adım 1: Kodu GitHub'a Push Et (Lokalde)

Terminalini aç ve şunu çalıştır:

```bash
cd C:\Users\telat\Documents\GitHub\DiffFit-3D

# .gitignore'u güncelle (büyük dosyaları git'e ekleme)
# data/ ve checkpoints/ zaten .gitignore'da olmalı

git add .
git commit -m "Full 3D pipeline ready for training"
git push origin main
```

> **Not**: `data/` ve `checkpoints/` klasörleri `.gitignore`'da olmalı çünkü çok büyükler (25+ GB).

---

## Adım 2: Veriyi Google Drive'a Yükle

Bu en uzun adım. Büyük dosyaları zip'le ve Drive'a yükle:

### Lokalde ZIP'le (PowerShell)
```powershell
cd C:\Users\telat\Documents\GitHub\DiffFit-3D

# Kişi görüntüleri (en büyük, ~4 GB)
Compress-Archive -Path "data\raw\images\*.jpg" -DestinationPath "drive_upload\images.zip"

# Processed verileri (pose, segment, densepose, agnostic)
Compress-Archive -Path "data\processed\poses" -DestinationPath "drive_upload\poses.zip"
Compress-Archive -Path "data\processed\segments" -DestinationPath "drive_upload\segments.zip"
Compress-Archive -Path "data\processed\densepose" -DestinationPath "drive_upload\densepose.zip"
Compress-Archive -Path "data\processed\agnostic" -DestinationPath "drive_upload\agnostic.zip"

# 3D giysi mesh'leri
Compress-Archive -Path "data\garments_3d\upper_body","data\garments_3d\lower_body","data\garments_3d\dresses","data\garments_3d\outerwear" -DestinationPath "drive_upload\garments_3d.zip"

# CSV dosyaları
Copy-Item "data\raw\*.csv" "drive_upload\"

# SMPL-X + VPoser
Compress-Archive -Path "checkpoints\pretrained" -DestinationPath "drive_upload\pretrained.zip"
```

### Google Drive'a Yükle
1. [drive.google.com](https://drive.google.com) aç
2. `DiffFit-3D` adında klasör oluştur
3. `drive_upload/` klasöründeki tüm zip'leri buraya sürükle-bırak

### Drive Yapısı:
```
Google Drive/
└── DiffFit-3D/
    ├── images.zip          (~4 GB)
    ├── poses.zip           (~500 MB)
    ├── segments.zip        (~200 MB)
    ├── densepose.zip       (~2 GB)
    ├── agnostic.zip        (~2 GB)
    ├── garments_3d.zip     (~5 GB)
    ├── pretrained.zip      (~170 MB)
    ├── train_pairs.csv
    ├── val_pairs.csv
    └── test_pairs.csv
```

---

## Adım 3: Colab Notebook'u Aç

1. [colab.research.google.com](https://colab.research.google.com) aç
2. Projedeki `notebooks/DiffFit3D_Train.ipynb` dosyasını yükle
   — VEYA —
   GitHub'dan aç: File → Open Notebook → GitHub → repo URL'ni yapıştır
3. Runtime → Change runtime type → **GPU** (T4/A100) seç
4. Her hücreyi sırayla çalıştır

---

## Adım 4: Eğitimi Başlat

Notebook hücreleri sırayla:
1. GPU kontrolü
2. Repo'yu clone et
3. Kütüphaneleri kur
4. Drive'ı mount et, verileri aç
5. SMPL-X parametre çıkarımı
6. 3D giysi render
7. Eğitim başlat

Detaylar notebook'un içinde.
