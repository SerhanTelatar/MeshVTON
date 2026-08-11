# İki eğitim rejimi — hangisi, neden, nasıl geri getirilir

Bu depo **tek kod tabanı**dır; iki farklı eğitim rejimi üretmiştir. Kod çatallanmadı,
fark yalnızca veri ve bayraklardadır. Tez sonuçları **A rejimine** dayanır.

## A — TEXTURE'LI (tezin dayandığı rejim)

- Checkpoint: `<Drive>/v2_outputs/stage1_july/ckpt_004000.pt` (2026-07-06, 4000 adım)
- Appearance ref: **texture'lı** (giysinin gerçek renk/deseni)
- Eğitim verisi: %70 gerçek VITON-HD + %30 sentetik çok-görüş
- Golden set: texture'lı giysiler (`manifest.JULY.json`)
- Çıkarım: `--use-texture`, `hang_pad=-0.12`, `garment_scale=1.25`

Sonuç (n=50): geo_iou 0.5234 → **0.7005**, mesh ayırt ediciliği +0.0778 → **+0.2205**,
kontrol dalının payı **%65**. Eğitimsiz baseline'ın ayırt ediciliği +0.0046 (≈ sıfır).
Çıktılar fotogerçekçi ve renkli; sınırlılık: giysi *kimliği* gevşek takip ediliyor.

## B — DOKUSUZ (2026-08 rejimi)

- Checkpoint: `<Drive>/v2_outputs/stage1/final.pt` (2026-08-10, 1000 adım)
- Appearance ref: `force_textureless` ile **her zaman düz gri**
- Eğitim verisi: yalnız sentetik (VITON-HD bilinçli olarak çıkarıldı)
- Golden set: `manifest.AUG.json`

Aynı sette: geo_iou 0.6072, ayırt edicilik +0.1368, kontrol payı %41.
**Her metrikte A'nın altında.** "Görünüm yolunu kaldırmak geometriyi güçlendirir"
hipotezi bu ölçümle çürüdü. Çıktılar gri ve düz.

## Kodda farkı yaratan noktalar

| konu | A (texture'lı) | B (dokusuz) | nerede |
|---|---|---|---|
| appearance ref | `use_texture=True` | varsayılan `False` | `builder.py::build_conditioning` |
| sentetik giysi seçimi | yalnız texture'lı | hepsi | `synth/generate.py::discover_assets` |
| gerçek veri | VITON-HD karışık | yok | `colab_pipeline.py` (vitonhd aşaması çıkarıldı) |
| golden set | texture'lı giysiler | herhangi | `build_golden_set.py --garment-ids` |

`use_texture` varsayılanı **False** kalır (proje kuralı). A rejimini koşmak için
bayrağı açıkça vermek gerekir — `eval_checkpoint.py --use-texture`,
`MeshVTON_inference_stage1.ipynb`.

## Yeniden eğitim gerekirse (A rejimi)

Kod çatallamaya gerek yok, üç değişiklik yeter:

1. `generate_synthetic.py` texture'lı giysilerle koşsun (texture filtresi geri) ve
   `build_conditioning(..., use_texture=True)` geçsin — GT ile ref aynı görünümde
   olmalı, yoksa akış-eşleme kaybı koşullu ortalamaya düşer (2026-08-09 dersi).
2. `colab_pipeline.py`'ye VITON-HD aşaması geri eklensin (`preprocess_vitonhd.py`
   script olarak duruyor).
3. `DATA_VERSION` yükseltilsin — sentetik veri yeniden üretilecek (~3 saat).

Maliyet: ~3 sa veri + ~11 sa eğitim (4000 adım, 10 sn/adım A100).

## Silinmemesi gerekenler

- `stage1_july/ckpt_004000.pt` — tezin dayandığı TEK checkpoint. Çöp kutusundan
  kurtarıldı; `stage1/` altına KOYMA, eğitim rotasyonu (`keep_last=2`) siler.
- `manifest.JULY.json` ve `manifest.AUG.json` — iki rejimin golden set'leri.
- `eval_results/` — tüm ölçümler ve figürler.