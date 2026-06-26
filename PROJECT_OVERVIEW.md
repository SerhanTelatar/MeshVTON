# MeshVTON — Proje Genel Bakış (Türkçe Özet)

> **Multi-View Virtual Try-On via 3D Garment Mesh Conditioning in Latent Diffusion Models**
>
> Bu doküman, MeshVTON projesinin tamamını — amacını, mimarisini, veri akışını, eğitim/çıkarım
> stratejisini, çalışma ortamını ve güncel durumunu — uçtan uca özetler. Kod tabanı, hafıza
> notları ve mevcut dokümantasyon esas alınarak yazılmıştır.

---

## 1. Proje Bir Cümlede

MeshVTON, bir **2D insan fotoğrafı** ile bir **3D giysi mesh'ini (`.obj`)** girdi olarak alıp,
giysiyi o kişinin tahmin edilen pozuna göre giydirilmiş şekilde, **herhangi bir kamera açısından
(ön / yan / arka) geometrik olarak tutarlı** biçimde üreten bir sanal giyinme (virtual try-on)
pipeline'ıdır. Çıktı, giysiyi giymiş kişinin 2D fotorealistik görüntüsüdür.

Temel taşıyıcısı, önceden eğitilmiş **IDM-VTON (SDXL tabanlı Latent Diffusion)** modelidir.
Projenin **özgün katkısı (novelty)**, bu dondurulmuş omurgaya 3B geometrik koşullandırma enjekte
eden **ControlNet3D** modülüdür.

---

## 2. Neden Var? (Motivasyon ve Özgünlük)

Klasik 2D virtual try-on yöntemleri tek bir düz giysi fotoğrafına dayanır ve şu sorunları yaşar:

- **Ön-yüz önyargısı** — Ağ giysiyi yalnızca önden gördüğü için, kişi arkaya dönse bile giysinin
  ön yüzünü çizmeye devam eder.
- **Perspektif tutarsızlığı** — Yan/arka görünümler "çıkartma" gibi yapay durur.
- **Vücut tipine kırılganlık** — 2D warping farklı vücut tiplerine genelleyemez.

MeshVTON bunu **gerçek 3B giysi mesh'leri** kullanarak çözer:

| Avantaj | Açıklama |
|---------|----------|
| 🔄 Arka & yan görünüm | Mesh herhangi bir açıdan render edilir; halüsinasyon gerekmez |
| 📐 Geometrik doğruluk | 3B mesh, her vücut tipinde doğru oranları garanti eder |
| 🎭 Normal & derinlik haritaları | Diffusion modeline ControlNet3D üzerinden 3B yapısal ipuçları verir |
| 💡 Fiziksel ışıklandırma | Phong gölgeleme gerçekçi gölge/parlama üretir |
| 🧠 IDM-VTON omurgası | Önceden eğitilmiş SDXL try-on modeli SOTA üretim kalitesi sağlar |

**Paradigma karşılaştırması:**

```
Klasik 2D try-on:  2D giysi fotoğrafı → warp → yapıştır → sonuç (yalnız ön)

MeshVTON:          3B giysi mesh'i → SMPL-X giydirme → PyTorch3D render
                          │                                       │
                   GarmentNet (dondurulmuş)            ControlNet3D (eğitilebilir)
                          │                                       │
                          └────►  TryonNet (dondurulmuş)  ◄───────┘
                                         │
                                  VAE decode → sonuç (her açıdan)
```

---

## 3. Yüksek Seviyeli Pipeline

| Aşama | Bileşen | Açıklama |
|-------|---------|----------|
| **Girdi** | İnsan görüntüsü + 3B giysi mesh'i | 2D fotoğraf + `.obj` giysi dosyası |
| **Vücut Tahmini** | SMPL-X Estimator | 3B vücut şekli (β), poz (θ), eklemler |
| **Giysi Giydirme** | Garment Draper | 3B mesh'i SMPL-X vücut üzerine sarar |
| **3B Render** | PyTorch3D Renderer | RGB render + normal harita + derinlik haritası |
| **Agnostic Üretimi** | DWPose + ATR + AgnosticMaskGenerator | Poz + segmentasyon → giysiden arındırılmış kişi |
| **Giysi Kodlama** | GarmentNet + IP-Adapter (CLIP Vision) | Self-attn füzyonu ve cross-attention için özellikler |
| **3B Koşullandırma** | **ControlNet3D** (özgün) | 9 kanal (RGB+normal+derinlik) → çok ölçekli residual'lar |
| **Omurga** | IDM-VTON (SDXL) | Dondurulmuş TryonNet + GarmentNet, cross-attention'lı |
| **Çıktı** | SDXL VAE Decoder + Post-Processing | Fotorealistik try-on sonucu |

---

## 4. Eğitim Stratejisi — Yalnızca ControlNet3D

Projenin verimlilik anahtarı: **omurganın tamamı dondurulur, sadece ControlNet3D eğitilir.**

| Bileşen | Parametre | Durum |
|---------|-----------|-------|
| TryonNet (SDXL UNet) | ~2.6B | ❄️ Dondurulmuş |
| GarmentNet (SDXL UNet) | ~2.6B | ❄️ Dondurulmuş |
| VAE (SDXL AutoencoderKL) | ~85M | ❄️ Dondurulmuş |
| CLIP Vision + Text Encoders | ~1.8B | ❄️ Dondurulmuş |
| IP-Adapter Resampler | ~50M | ❄️ Dondurulmuş |
| **ControlNet3D (özgün)** | **~350–400M** | ✅ **Eğitilebilir** |

Toplam parametrenin yalnızca **~%5–7'si** eğitilir. Checkpoint sadece ControlNet3D ağırlıklarını
içerir (~485MB), `checkpoints/meshvton/` altında.

---

## 5. 3B Pipeline (Özgün Katkı) — Detay

### 5.1 SMPL-X Vücut Tahmini
İnsan fotoğrafı SMPL-X parametrelerine regresyon edilir:
- `betas (β)` (B,10) — vücut şekli
- `body_pose (θ)` (B,63) — 21 eklem × 3 eksen-açı
- `global_orient` (B,3), `transl` (B,3)
- `vertices` (B,10475,3), `joints` (B,127,3), `faces` (20908,3)

> Mevcut implementasyon basit bir ResNet-tarzı regresördür (placeholder / `SimpleSMPLXRegressor`).
> Üretimde PyMAF-X veya ExPose ile değiştirilmesi hedeflenir. **Yeni kişiden SMPL-X poz tahmini
> henüz eğitilmemiştir** — bu yüzden arka görünüm / poza tam uyumlu giydirme hâlâ iyileştirme
> bekliyor (bkz. Bölüm 11).

### 5.2 Garment Draper
3B giysi mesh'ini SMPL-X vücuduna sarar.

> **Önemli düzeltme:** Nöral `GarmentDraper` ağı eğitilmemişti ve mesh'i bir "blob"a çöküyordu.
> Bunun yerine **geometrik hizalama** (`_geometric_align`) kullanıldı: CLOTH3D'nin Z-up eksenini
> Y-up'a çevir `(x,y,z) → (x,z,-y)`, orijine ortala, vücut bounding-box'ına ölçekle. Ayrıca
> `SMPLXEstimator.get_body_mesh` çağrısında `.detach()` gerekti.

### 5.3 PyTorch3D Diferansiyellenebilir Render
Sarılmış mesh'i istenen kamera açısından render eder:
- Kameralar: `FoVPerspectiveCameras` (dist=2.7, elev=0, azim ∈ [0,360°])
- Işık: `PointLights` + `SoftPhongShader` (ambient + diffuse + specular)
- Çıktılar: RGB (B,3,H,W), Normal (B,3,H,W), Depth (B,1,H,W → 3 kanala yayılır)

| `azim` | Görünüm |
|--------|---------|
| 0° | Ön |
| 90° / 270° | Yan profil |
| 180° | **Arka — 2D yöntemlerin yapamadığı** |

> `render_garment`, yalnızca kişide SMPL-X **ve** karşılık gelen giysinin mesh'i bulunan çiftleri
> render eder; `garment_id`, mesh'in bulunduğu üst klasör adıdır. Çiftler bu kesişime filtrelenir.

---

## 6. ControlNet3D — Özgün Modül

ControlNet3D, 3B render çıktılarını (RGB + normal + derinlik) çok ölçekli residual bağlantılarla
TryonNet'e enjekte eder.

### 6.1 Koşullandırma Girdisi
```
conditioning_3d : (B, 9, H, W)
  ├── RGB render   (B,3,H,W)
  ├── Normal harita (B,3,H,W)
  └── Derinlik     (B,3,H,W)  (1→3 kanala yayılmış)
```

### 6.2 Conditioning Encoder
9 kanallı girdiyi 8× downsample ederek UNet'in temel kanal genişliğine (320) indirir:
`Conv(9→16)→…→Conv(256→320, stride=2)` (her adımda SiLU). Çıktı: `(B,320,H/8,W/8)`.
İmplementasyon: [src/models/controlnet_3d.py](src/models/controlnet_3d.py).

### 6.3 Encoder Blokları (SDXL UNet'i yansıtır)
SDXL kademelerini yansıtan ResBlock + Downsample dizisi; her blok çıktısı bir `ZeroConv`'tan geçip
bir residual üretir (toplam 12 down residual + 1 mid residual). `ControlNet3DResBlock` =
`GroupNorm → SiLU → Conv3×3 → (+timestep_embed) → GroupNorm → SiLU → Conv3×3 → skip`.

### 6.4 Sıfır-İnisiyalizasyon (Zero-Init)
Her `ZeroConv`, ağırlık ve bias'ı **sıfıra** ayarlanmış 1×1 konvolüsyondur:
- Başlangıçta her residual = 0 → dondurulmuş TryonNet hiç bozulmaz (`h + 0 = h`)
- Eğitim ağırlıkları sıfırdan kademeli uzaklaştırır
- Standart ControlNet hilesi (Zhang ve diğ., 2023)

### 6.5 TryonNet'e Enjeksiyon
```python
down_block_additional_residuals = controlnet_residuals[:-1]
mid_block_additional_residual    = controlnet_residuals[-1]
```
Her encoder kademesinin ve mid bloğun çıktısına `h_i ← h_i + r_i` olarak eklenir.

---

## 7. IDM-VTON Omurgası (Dondurulmuş)

### 7.1 TryonNet — Ana Denoising Omurgası
`yisol/IDM-VTON`'dan yüklenen, "hacked" SDXL UNet (`src/idm_vton/unet_hacked_tryon.py`).
**13 kanallı girdi** (IDM-VTON'un imza yeniliği):
```
unet_in = cat([
    noisy_latent / zt,       # (B,4) gürültülü kişi latent'i
    mask,                    # (B,1) inpaint maskesi
    masked_image_latents,    # (B,4) agnostic kişi (VAE)
    pose_img / densepose,    # (B,4) densepose (VAE)
], dim=1)  → (B,13,h,w)
```

> ⚠️ **Kritik mimari kararı (2026-06):** Kanal sıralaması IDM-VTON'un dondurulmuş omurgasının
> beklediği `[noise(4), mask(1), masked_image(4), pose/densepose(4)]` düzeniyle **birebir aynı**
> olmalıdır. Bu sıralama [src/idm_vton/tryon_pipeline.py](src/idm_vton/tryon_pipeline.py)'deki
> denoising döngüsünde tanımlıdır. (Detay için bkz. Bölüm 10.)

### 7.2 GarmentNet — Giysi Referans Özellik Çıkarıcı
İkinci bir SDXL UNet (`unet_hacked_garmnet`). `cloth_lat`'tan `reference_features` üretir; bunlar
TryonNet'in **self-attention füzyonu** yoluyla enjekte edilir (`garment_features` parametresi).

### 7.3 IP-Adapter — Giysi Görsel Özellikleri
`CLIPVisionModelWithProjection` → penultimate (`hidden_states[-2]`) → Resampler
(dim=1280, depth=4, heads=20, num_queries=16, output=2048) → TryonNet cross-attention.
Eğitimde Resampler, `unet.encoder_hid_proj` içinde yer alır ve dondurulmuştur.

### 7.4 Text Encoders & VAE
- SDXL çift text encoder: `prompt_embeds (B,77,2048)` + pooled `(B,1280)`; boş prompt'ta bile
  SDXL formatı gerektiği için kullanılır.
- VAE (SDXL AutoencoderKL), `scaling_factor = 0.13025`. Aynı VAE üç akışı kodlar: kişi, agnostic,
  giysi/render.

---

## 8. 2B Ön İşleme — Yalnızca Agnostic Görüntü İçin

Mevcut pipeline poz haritalarını/DensePose'u modele doğrudan beslemez (densepose `pose_img`
kanalı hariç). 2B ön işleme zinciri esas olarak **giysiden arındırılmış (agnostic) kişi görüntüsünü**
üretmek için vardır:

```
Person → PoseEstimator (DWPose) → keypoints (18,3)
       → HumanSegmentation (ATR) → segments (H,W)
       → AgnosticMaskGenerator(image, segments, keypoints) → agnostic image
```

Ön işleme scriptleri:
- [src/data/preprocessing/extract_pose.py](src/data/preprocessing/extract_pose.py) — DWPose keypoints
- [src/data/preprocessing/extract_segment.py](src/data/preprocessing/extract_segment.py) — ATR body parsing
- [src/data/preprocessing/build_agnostic.py](src/data/preprocessing/build_agnostic.py) — agnostic kişi
- [src/data/preprocessing/extract_smplx.py](src/data/preprocessing/extract_smplx.py) — SMPL-X parametre + mesh
- [src/data/preprocessing/render_garment.py](src/data/preprocessing/render_garment.py) — RGB + normal + derinlik

---

## 9. Veri Kümesi ve Veri Akışı

| Veri Kümesi | İçerik | Kullanım |
|-------------|--------|----------|
| [VITON-HD](https://github.com/shadow2496/VITON-HD) | İnsan fotoğrafları | Eğitim görüntüleri |
| [CLOTH3D](https://chalearnlap.cvc.uab.cat/) | 3B giysi mesh'leri (OBJ + doku) | 3B giysi varlıkları |
| [SMPL-X](https://smpl-x.is.tue.mpg.de/) | Vücut model parametreleri | 3B vücut tahmini |

**`MeshVTONDataset`** ([src/data/dataset.py](src/data/dataset.py)) örnek başına ürettiği tensörler
(hepsi `(height, width)`):

| Anahtar | Şekil | Açıklama |
|---------|-------|----------|
| `person` | (3,H,W) [-1,1] | Hedef: giysiyi giymiş kişi |
| `masked_image` | (3,H,W) [-1,1] | Agnostic kişi |
| `mask` | (1,H,W) [0,1] | İnpaint bölgesi (1 = giysi alanı) |
| `pose_img` | (3,H,W) [-1,1] | Kişinin densepose'u |
| `cloth` | (3,H,W) [-1,1] | Kişinin pozuna render edilmiş 3B giysi |
| `conditioning_3d` | (9,H,W) [-1,1] | render(3)+normal(3)+derinlik(3) → ControlNet3D |

Beklenen dizin yerleşimi:
```
{data_root}/raw/images/{person_id}.jpg
{data_root}/processed/agnostic/{person_id}.jpg
{data_root}/processed/densepose/{person_id}.jpg|png
{data_root}/processed/renders_3d/{person_id}_{garment_id}.png
{data_root}/processed/normal_maps/{person_id}_{garment_id}.png
{data_root}/processed/depth_maps/{person_id}_{garment_id}.png
```

> `renders_3d` (ölçeklenmiş/render edilmiş giysi) hem GarmentNet girdisi (`cloth`) hem de
> ControlNet3D koşullandırması olarak görev yapar.

---

## 10. Eğitim — Gerçek IDM-VTON Pipeline Üzerinde (Phase 2)

Eğitim, [scripts/train_meshvton.py](scripts/train_meshvton.py) ile **dondurulmuş gerçek IDM-VTON
pipeline'ı** üzerinde, IDM-VTON'un tam forward kontratıyla yapılır. Adım başına (CFG yok):

```
z0          = vae(person) · scaling                      # hedef latent
zt          = scheduler.add_noise(z0, eps, t)
unet_in     = cat([zt, mask, vae(masked_image), vae(pose_img)])   # 13 kanal
ref_feats   = GarmentNet(vae(cloth), t, text_c)                   # self-attn referansı
image_embeds= unet.encoder_hid_proj(image_encoder(clip(cloth))[-2])  # IP-Adapter
residuals   = ControlNet3D(conditioning_3d, t)            # 12 down + 1 mid (EĞİTİLEBİLİR)
eps_pred    = unet(unet_in, t, text, added_cond_kwargs,
                   down/mid residuals, garment_features=ref_feats)
loss        = MSE(eps_pred, eps)
```

Yalnızca `ControlNet3D.parameters()` gradyan alır. Çalıştırma:
```bash
python scripts/train_meshvton.py --data_root data --pairs data/raw/train_pairs.csv
```

### Hiperparametreler (özet)
| Parametre | Değer |
|-----------|-------|
| Çözünürlük | 512 × 384 (train script) / 512 × 512 (config) |
| Latent | 4 × 64 × 64 |
| Optimizer | AdamW (lr=1e-4 script / 1e-5 config), weight_decay=0.01 |
| LR schedule | cosine + 500 adım warmup |
| Precision | bf16 (script) / fp16 (config) |
| Grad accumulation | 16 (efektif batch 16) |
| Max grad norm | 1.0 |
| Diffusion steps | 1000 (train, DDPM) / 50 (infer, DDIM) |
| EMA decay | 0.9999 (yalnız ControlNet3D) |

> Loss şu an saf **MSE(ε̂, ε)**'dir. `TryOnLoss` içinde ek VGG perceptual / LPIPS / adversarial / KL
> terimleri tanımlıdır ama varsayılan forward yalnız gürültü MSE'sini döndürür.

---

## 11. ⚠️ Önemli Mimari Karar: Custom Pipeline Terk Edildi

> Bu, kod tabanında **gizli ama kritik** bir gerçektir ve `docs/ARCHITECTURE.md`'nin bazı
> kısımları hâlâ eski custom pipeline'a atıf yapar.

- El yazımı [src/models/tryon_pipeline.py](src/models/tryon_pipeline.py) (custom
  `TryOnPipeline.forward`/`generate`) **BOZUK ve sonuç üretimi için terk edilmiştir.**
  - Custom 13 kanal düzeni `[noise, agnostic, mask, garment]`, IDM-VTON'un dondurulmuş
    backbone'unun beklediği `[noise(4), mask(1), masked_image(4), pose/densepose(4)]` düzeniyle
    uyuşmuyordu → dondurulmuş `conv_in` karıştı → saf gürültü çıktısı.
- **Çalışan yaklaşım:** [src/idm_vton/tryon_pipeline.py](src/idm_vton/tryon_pipeline.py) içindeki
  gerçek `StableDiffusionXLInpaintPipeline` (doğrulandı: temiz try-on üretir).
- **Durum (2026-06-17):** Tüm aşamalar uçtan uca doğrulandı; ilk temiz MeshVTON sonucu alındı —
  3B bermuda şort mesh'inin şekli kişiye aktarıldı, gürültü yok.

**Kalan işler / bilinen kısıtlar:**
- Tüm veri kümesinin render edilmesi + kalite için daha uzun eğitim.
- Yeni kişiden **SMPL-X poz tahmini eğitilmemiş** (`SimpleSMPLXRegressor`) → poza tam uyumlu /
  arka görünüm giydirme hâlâ iyileştirme bekliyor.

---

## 12. Çıkarım (Inference)

### 12.1 3B-farkında mod (özgün)
```
1. Ön işleme: agnostic görüntü
2. 3B pipeline: SMPL-X → mesh yükle → drape → render(view_angle)
   conditioning_3d = cat(rgb, normal, depth)  # (B,9,H,W)
3. Giysinin ön-render'ı garment_image olarak kullanılır (GarmentNet/IP-Adapter doğru geometriyi görür)
4. DDIM döngüsü (50 adım) + her adımda ControlNet3D residual'ları
5. VAE decode + post-process
```

### 12.2 Çok-açılı üretim
```bash
python scripts/inference.py --person p.jpg --garment g.obj --view_angle 0    # ön
python scripts/inference.py --person p.jpg --garment g.obj --view_angle 90   # yan
python scripts/inference.py --person p.jpg --garment g.obj --view_angle 180  # arka (2D yapamaz)
```

### 12.3 Classifier-Free Guidance
`ε̂ = ε_uncond + w·(ε_cond − ε_uncond)`, `w = 7.5`. Post-processing: yüz restorasyonu (CodeFormer),
kenar yumuşatma, renk düzeltme, opsiyonel SynthID watermark.

---

## 13. Çalışma Ortamı (Colab)

- **Donanım:** RTX PRO 6000 Blackwell, 102 GB. Python 3.12, torch 2.11.0+cu128, CUDA 12.8, nvcc mevcut.
- **Pinlenmiş sürümler (IDM-VTON uyumu):**
  `diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0 huggingface_hub==0.20.3 peft==0.7.1`
  - `huggingface_hub 0.20.3`: yenisi diffusers 0.25'in import ettiği `cached_download`'u kaldırıyor.
  - `peft 0.7.1`: yenisi accelerate 0.25'te olmayan `clear_device_cache`'i import ediyor.
- **detectron2 (densepose için):** torch 2.11 için hazır wheel yok → kaynaktan derle:
  `FORCE_CUDA=1 pip install 'git+https://github.com/facebookresearch/detectron2.git'` (derlenip
  çalışıyor; iopath'i 0.1.9'a düşürür).
- **Ön işleme (densepose + mask):** `yisol/IDM-VTON` GitHub reposu klonlanır (`apply_net.py`,
  `preprocess/`, `utils_mask.get_mask_location`). LFS pull kota nedeniyle başarısız → model dosyaları
  doğrudan indirilir (densepose `model_final_162be9.pkl`, humanparsing `.onnx`, openpose
  `body_pose_model.pth`). Ayrıca `pip install av onnxruntime-gpu`.
- **`src` isim çakışması:** Hem MeshVTON hem IDM-VTON-official `src/` paketine sahip. IDM-VTON
  pipeline'ı kendi kopyamız `src.idm_vton.*` üzerinden, `/content/MeshVTON` sys.path'te ilk sırada
  (önce cache'lenmiş `src`'yi sys.modules'tan temizleyerek) import edilir.
- **Notebook'lar:** Inference notebook modülleri kernel'a import edildiğinden, `git pull` sonrası kod
  değişikliği kernel restart / modül reload gerektirir (train.py subprocess olarak çalıştığından
  ondan etkilenmez).

---

## 14. Proje Yapısı

```
MeshVTON/
├── configs/
│   ├── train.yaml / inference.yaml
│   └── data/              # dataset.yaml, preprocessing.yaml
├── src/
│   ├── idm_vton/          # ✅ GERÇEK çalışan IDM-VTON pipeline (hacked UNet'ler)
│   │   ├── tryon_pipeline.py       # StableDiffusionXLInpaintPipeline (çalışan)
│   │   ├── unet_hacked_tryon.py    # TryonNet (13-ch, residual + garment_features)
│   │   └── unet_hacked_garmnet.py  # GarmentNet
│   ├── models/            # controlnet_3d.py (özgün) + tryon_pipeline.py (BOZUK/terk)
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
├── docs/                  # ARCHITECTURE.md (kısmen eski), DATASET_GUIDE.md, diyagram
├── tests/                 # test_pipeline.py, test_warping.py
├── data/                  # veri kümeleri (gitignored)
└── checkpoints/           # ağırlıklar (gitignored); meshvton/ → ControlNet3D-only ~485MB
```

> **Not:** `src/idm_vton/` (çalışan) ile `src/models/tryon_pipeline.py` (terk edilmiş) ayrımı
> kritiktir — yeni geliştirme `src/idm_vton/` üzerinden yapılmalıdır.

---

## 15. Anahtar Hiperparametre Özeti

| Parametre | Değer | Not |
|-----------|-------|-----|
| Görüntü çözünürlüğü | 512×512 / 512×384 | config / train script |
| Latent | 4×64×64 | SDXL VAE |
| VAE scaling | 0.13025 | SDXL (SD1.5: 0.18215) |
| UNet kanal çarpanları | (1,2,4,4) | 320→640→1280→1280 |
| ControlNet3D girdi | 9 kanal | RGB(3)+Normal(3)+Depth(3) |
| TryonNet girdi | **13 kanal** | noise(4)+mask(1)+masked(4)+pose(4) |
| IP-Adapter token | 16 | dim=2048 |
| Diffusion steps | 1000 train / 50 infer | DDPM / DDIM |
| Guidance | 7.5 | CFG |
| Eğitilebilir param | ~350–400M | ~%5–7 |

---

## 16. Güncel Durum & Yol Haritası

**✅ Tamamlanan (2026-06-17 itibarıyla):**
- Phase 1: Pipeline `__call__` içinde ControlNet3D residual enjeksiyonu (None-safe, 2D yolu bozulmaz).
- Phase 2: `MeshVTONDataset` + `train_meshvton.py` ile dondurulmuş gerçek pipeline üzerinde
  ControlNet3D eğitimi (IDM-VTON'un tam 13-ch forward'ı, eğitilmiş Resampler, GarmentNet referans
  özellikleri, gürültü MSE'si).
- Geometrik giydirme/render düzeltmeleri (blob sorunu çözüldü).
- İlk temiz uçtan uca MeshVTON sonucu.

**🔜 Kalan / Hedefler:**
- Tüm veri kümesini render et + kalite için daha uzun eğitim.
- Gerçek SMPL-X poz tahmin modülü (PyMAF-X / ExPose) entegrasyonu → poza/arkaya doğru giydirme.
- Auxiliary loss terimlerinin (LPIPS, perceptual) etkinleştirilmesi.
- `docs/ARCHITECTURE.md`'nin custom pipeline atıflarından arındırılıp `src/idm_vton/` çalışan
  yoluna güncellenmesi.
- Video try-on yolu (`src/video/`, `src/inference/video_tryon.py`) henüz deneysel.

---

## 17. Lisans & Atıf

Apache 2.0. 
```bibtex
@software{MeshVTON2025,
  title={MeshVTON: Geometry-Aware Virtual Try-On},
  author={Serhan Telatar},
  year={2025},
  url={https://github.com/SerhanTelatar/MeshVTON}
}
```
