# MeshVTON: Dataset Hazırlama Rehberi

## 📥 1. İndirilecek Veriler

MeshVTON tam 3D pipeline'ı için **3 tip veri** gereklidir:

---

### 1.1 — Kişi Görüntüleri (2D)

Herhangi bir virtual try-on datasetinden kişi görüntüleri:

| Dataset | İndirme | Boyut | Not |
|---------|---------|-------|-----|
| **VITON-HD** | [GitHub](https://github.com/shadow2496/VITON-HD) | ~12 GB | En yaygın benchmark |
| **DressCode** | [GitHub](https://github.com/aimagelab/dress-code) | ~50 GB | Üst/Alt/Elbise |
| **DeepFashion** | [mmlab.ie.cuhk.edu.hk](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) | ~30 GB | Büyük ölçekli |

**Adım**: Datasetlerin web sayfasına git → Academic kullanım için form doldur → İndirme linkini al

---

### 1.2 — 3D Giysi Mesh'leri

| Kaynak | İndirme | Format | Not |
|--------|---------|--------|-----|
| **CLOTH3D** | [cloth3d.github.io](https://chalearnlap.cvc.uab.cat/dataset/38/description/) | OBJ | Sentetik, SMPL uyumlu |
| **Deep Fashion3D** | [GitHub](https://github.com/kv2000/DeepFashion3D) | OBJ/PLY | Gerçek taranmış |
| **ClothesNet** | [clothesnet.github.io](https://clothesnet.github.io/) | OBJ | 3000+ giysi |
| **Sketchfab** | [sketchfab.com](https://sketchfab.com/search?q=clothing&type=models) | GLB/OBJ | Ücretsiz modeller mevcut |
| **TurboSquid** | [turbosquid.com](https://www.turbosquid.com/Search/3D-Models/free/clothing) | OBJ/FBX | Ücretsiz/ücretli |
| **CGTrader** | [cgtrader.com](https://www.cgtrader.com/free-3d-models/clothes) | OBJ | Ücretsiz modeller |

**En iyi başlangıç**: **CLOTH3D** — SMPL beden modeli ile uyumlu, etiketlenmiş, sentetik

---

### 1.3 — SMPL-X Beden Modeli

| Dosya | İndirme | Not |
|-------|---------|-----|
| **SMPL-X model** | [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) | Kayıt gerekli (akademik) |
| Gerekli dosyalar | `SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz` | ~300 MB |

**Adım**: smpl-x.is.tue.mpg.de → Register → Download SMPL-X → `SMPLX_*.npz` dosyalarını al

---

## 📁 2. Dosya Yerleşimi

İndirdiğin dosyaları aşağıdaki yapıya göre yerleştir:

```
MeshVTON/
├── checkpoints/
│   └── pretrained/
│       └── smplx/                          ← SMPL-X MODEL DOSYALARI
│           ├── SMPLX_NEUTRAL.npz           ← İndir: smpl-x.is.tue.mpg.de
│           ├── SMPLX_MALE.npz
│           └── SMPLX_FEMALE.npz
│
├── data/
│   ├── raw/
│   │   ├── images/                         ← KİŞİ GÖRÜNTÜLERİ (2D)
│   │   │   ├── 00001_00.jpg                   VITON-HD'den
│   │   │   ├── 00002_00.jpg
│   │   │   └── ...
│   │   ├── train_pairs.csv                 ← EĞİTİM ÇİFTLERİ
│   │   ├── val_pairs.csv
│   │   └── test_pairs.csv
│   │
│   ├── garments_3d/                        ← 3D GİYSİ MESH'LERİ
│   │   ├── upper_body/
│   │   │   ├── tshirt_001/
│   │   │   │   ├── mesh.obj                   3D geometri
│   │   │   │   ├── texture.png                UV doku haritası
│   │   │   │   └── metadata.json              Kategori ve malzeme bilgisi
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
│   └── processed/                          ← OTOMATİK ÜRETİLİR
│       ├── poses/                             Poz keypoints
│       ├── segments/                          Segmentasyon
│       ├── densepose/                         IUV haritaları
│       ├── agnostic/                          Agnostik maskeler
│       ├── smplx_params/                      SMPL-X beden parametreleri
│       ├── smplx_meshes/                      SMPL-X beden mesh'leri (.obj)
│       ├── renders_3d/                        Render edilmiş giysi görüntüleri
│       ├── normal_maps/                       Normal haritaları
│       └── depth_maps/                        Derinlik haritaları
```

---

## 📋 3. CSV Dosya Formatı

`train_pairs.csv` dosyası kişi ve 3D giysi mesh'ini eşleştirir:

```csv
person_id,garment_id
00001_00,tshirt_001
00002_00,shirt_002
00003_00,tshirt_001
00001_00,dress_001
...
```

- **person_id**: `data/raw/images/` altındaki dosya adı (uzantısız)  
- **garment_id**: `data/garments_3d/*/` altındaki klasör adı

---

## 📋 4. Giysi Metadata Formatı

Her giysi klasöründe `metadata.json`:

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

## 🚀 5. Ön-İşleme Komutları

İndirmeleri yerleştirdikten sonra tüm ön-işlemeyi çalıştır:

```bash
# 1. Tüm 2D ön-işleme (poz, segmentasyon, DensePose, agnostik)
python scripts/preprocess_dataset.py --steps pose segment densepose agnostic

# 2. SMPL-X beden parametreleri çıkar (YENİ — 3D)
python src/data/preprocessing/extract_smplx.py \
    --image_dir data/raw/images \
    --output_dir data/processed/smplx_params \
    --model_dir checkpoints/pretrained/smplx \
    --mesh_dir data/processed/smplx_meshes \
    --save_mesh

# 3. 3D giysileri render et (YENİ — 3D)
python src/data/preprocessing/render_garment.py \
    --garments_dir data/garments_3d \
    --smplx_params_dir data/processed/smplx_params \
    --output_dir data/processed/renders_3d \
    --normal_maps_dir data/processed/normal_maps \
    --depth_maps_dir data/processed/depth_maps \
    --resolution 512
```

---

## ⚙️ 6. Gerekli Kütüphaneler (3D Bağımlılıklar)

```bash
# 3D pipeline için ek paketler
pip install smplx trimesh pytorch3d

# PyTorch3D CUDA derleme gerektirebilir:
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

---

## 🔍 7. Minimum Başlangıç Veri Seti

Hızlı test için minimum gereksinim:

| Bileşen | Minimum | Önerilen |
|---------|---------|----------|
| Kişi görüntüleri | 10 | 1000+ |
| 3D giysi mesh'leri | 3 | 50+ |
| SMPL-X model | 1 (neutral) | 3 (neutral+male+female) |

---

## ❓ Sık Sorulan Sorular

**S: 3D mesh'im UV haritası içermiyor, sorun olur mu?**  
C: Hayır, sistem otomatik olarak vertex coloring'e düşer. Ancak UV doku daha iyi sonuç verir.

**S: Kendi 3D modelimi oluşturabilir miyim?**  
C: Evet! Blender, CLO3D veya Marvelous Designer ile giysi modeli oluşturup OBJ olarak export edebilirsin.

**S: SMPL-X olmadan çalışır mı?**  
C: Sistem placeholder mesh kullanır ama sonuçlar kötü olur. SMPL-X indirmek şiddetle tavsiye edilir.
