# MeshVTON v2

FLUX.1 Fill tabanlı, ekran-uzayı geometri koşullamalı, sentetik multi-view süpervizyonlu
virtual try-on. v1'in (IDM-VTON/SDXL) yerini alır; tasarım gerekçeleri ve fazlar için
plan dosyasına bakın. **v2 hiçbir zaman v1 `src/`'sinden import etmez** (bağımlılık
sürümleri uyumsuz — ayrı ortam, ayrı runtime).

## Kurallar (v1 dersleri)

1. **Parite:** Koşullama YALNIZ `meshvton2/conditioning/builder.py::build_conditioning`
   ile üretilir — eğitim ön-işleme, sentetik üretici ve inference aynı fonksiyonu çağırır.
   `tests/test_parity.py` bunu zorlar.
2. **Tek çözünürlük:** 768×1024 (`configs/base.yaml`). Başka çözünürlük yok.
3. **Kontrolde RGB yok:** geometri kanalları = normal + depth + silüet. Görünüm yalnız
   dokulu referanstan gelir (gri-render halüsinasyon dersi).
4. **Metrik yalanı yok:** hesaplanamayan metrik `None`/"n/a" döner, asla 0.0 değil.
5. **Notebook'ta mantık yok:** `notebooks/MeshVTON2.ipynb` üç hücrelik kabuktur.

## Durum

- [x] Faz 0 — eval harness + golden set altyapısı + parite sözleşmesi (builder stub)
- [~] Faz 1 — zero-shot taban çizgisi: kod hazır (`fill_spatial` + `kontext` varyantları),
      Colab koşusu bekliyor → `notebooks/MeshVTON2.ipynb`. Not: üçüncü varyant
      (eğitimsiz ref-token) Faz 4'e ertelendi — gerekçe `model/flux_tryon.py` docstring'inde.
- [~] Faz 2 — geometri hattı: kod tamam (pred_cam kamerası, LBS drape, ekran-uzayı
      render, gerçek builder — hepsi pyrender, pytorch3d gerekmez); Colab doğrulaması
      bekliyor (reprojection IoU ≥ 0.70 + drape QA sheet)
- [~] Faz 3 — sentetik multi-view veri: üretici hazır (`scripts/generate_synthetic.py`,
      dizin sözleşmesi testli); Colab'da küçük parti + QA bekliyor
- [ ] Faz 4 — Aşama-1 eğitim (tek görüş: LoRA + zero-init kontrol kolonları)
- [ ] Faz 5 — Aşama-2 eğitim (multi-view tutarlılık)
- [ ] Faz 6 — Blender veri v2 + inference sertleştirme

## Hızlı başlangıç

```bash
pip install -r v2/requirements.txt
python -m pytest v2/tests -q                       # sözleşme + metrik testleri
python v2/scripts/eval.py --self-check <img_dir>   # harness sıhhati
# Veri hazır olunca (Colab):
python v2/scripts/build_golden_set.py --vitonhd-test <test/image> --garments <garments_3d>
python v2/scripts/eval.py --pred-dir <tahminler>
```
