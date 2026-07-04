#!/usr/bin/env python3
"""MeshVTON v2 mimari diyagramı — koda sadık (code-truthful) üretim.

Kaynaklar: meshvton2/model/flux_tryon.py (kanal düzeni, LoRA, sampler),
model/control_embedder.py (zero-init), conditioning/* (kamera, drape, render).
Çalıştır: python v2/docs/draw_architecture.py  ->  v2/docs/architecture_v2.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# renk paleti (v1 diyagramıyla akraba)
C_FROZEN = "#dbe9f8"   # donuk backbone (mavi)
C_TRAIN = "#d9ead3"    # eğitilen (yeşil)
C_GEO = "#f4cccc"      # 3D/geometri (kırmızımsı)
C_APP = "#e6d5f0"      # görünüm (mor)
C_UTIL = "#eeeeee"     # yardımcı (gri)
C_IO = "#dce6f5"       # girdi/çıktı (açık mavi)
C_TOKEN = "#fff2cc"    # token şeridi (sarı)

fig, ax = plt.subplots(figsize=(17, 17.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 104)
ax.axis("off")


def box(x, y, w, h, text, fc, fs=10.5, weight="normal", ec="#666666", lw=1.2, style="round,pad=0.35"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)
    return (x + w / 2, y, y + h)  # cx, alt, üst


def arrow(x0, y0, x1, y1, color="#333333", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))


def section(x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc="none",
                                ec="#999999", lw=1.1, linestyle="--"))
    ax.text(x + 0.8, y + h - 1.6, label, fontsize=11, weight="bold", color="#555555", ha="left")


def route(points, color="#2e5f8a", lw=2.0):
    """Kutulara çarpmadan kenar koridorundan L-şekilli bağlantı (son bacak oklu)."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="round", zorder=1)
    arrow(points[-2][0], points[-2][1], points[-1][0], points[-1][1], color=color, lw=lw)


ax.text(50, 102.3, "MeshVTON v2: FLUX.1 Fill + Ekran-Uzayı Geometri + Referans Token'ları",
        ha="center", fontsize=19, weight="bold")

# ------------------------------- ÜST: çıktı ------------------------------- #
box(38, 97.2, 24, 2.9, "Try-On Sonucu (0° / 90° / 180° / 270°)", C_TRAIN, weight="bold")
box(38, 93.2, 24, 2.7, "Maske-dışı kompozit (kişi/arka plan korunur)", C_UTIL, fs=9.5)
box(38, 89.3, 24, 2.7, "FLUX VAE Decoder", C_FROZEN)
arrow(50, 92.2, 50, 93.0)
arrow(50, 96.1, 50, 97.0)

# --------------------------- ORTA: FLUX gövdesi --------------------------- #
section(8, 60.5, 84, 26.5, "GÖVDE — FLUX.1 Fill-dev Transformer (12B DiT, DONUK)")
box(11, 78.5, 37, 5.6,
    "ControlXEmbedder\nx_embedder (384ch, donuk)  +  control_proj (128ch)\nZERO-INIT → başlangıçta bit-eş stok Fill",
    C_TRAIN, fs=9.5)
box(52, 78.5, 37, 5.6,
    "LoRA  r=64  (attn q/k/v/out + FF)\n~200M eğitilebilir parametre\n(görev: referanstan giydir + geometri oku)",
    C_TRAIN, fs=9.5)
box(11, 70.5, 78, 5.2,
    "57 blok — tam dikkat: hedef ↔ referans ↔ metin token'ları\n"
    "T5/CLIP metin embed'leri sabit prompt için BİR KEZ hesaplanır (encoder'lar bellekte tutulmaz)",
    C_FROZEN, fs=9.5)
box(11, 63.0, 78, 5.0,
    "Rectified Flow — eğitim: logit-normal t + çözünürlük shift'i, kayıp YALNIZ hedef token'larda\n"
    "inference: Euler, 28 adım (FluxTryOnSampler = eğitim forward'ının aynası)",
    C_FROZEN, fs=9.5)
arrow(50, 86.9, 50, 89.1)  # gövde -> decoder yönü (yukarı)

# --------------------------- token dizisi şeridi --------------------------- #
box(8, 54.0, 50, 4.6,
    "HEDEF token'lar (frame 0):\n[ x_t 64 | masked 64 | mask 256 | kontrol 128 ]  = 512 kanal",
    C_TOKEN, fs=9.5)
box(61, 54.0, 31, 4.6,
    "REFERANS token'ları (frame 1):\n[ ref 64 | ref 64 | 0·256 | 0·128 ]",
    C_TOKEN, fs=9.5)
arrow(33, 58.6, 40, 60.7)
arrow(76.5, 58.6, 65, 60.7)

# ------------------------------ ALT: 3 kol ------------------------------ #
section(2, 11.8, 34, 39.2, "GEOMETRİ (ekran-uzayı)")
section(38, 11.8, 28, 39.2, "KİŞİ")
section(68, 11.8, 30, 39.2, "GÖRÜNÜM (appearance)")

# --- Kol A: geometri --- #
box(5, 44.0, 13, 3.6, "Kişi Fotoğrafı", C_IO, fs=9.5)
box(20, 44.0, 13, 3.6, "3D Giysi Mesh\n(.obj + texture)", C_IO, fs=9)
box(5, 38.2, 13, 4.2, "HMR2 (4D-Humans)\npred_cam + bbox\nARTIK ATILMIYOR", C_GEO, fs=8.7)
box(20, 38.2, 13, 4.2, "LBS yüzey bağlama\n(rest'te 1 kez, .npz cache)", C_GEO, fs=8.7)
box(5, 32.4, 13, 4.2, "SMPL-X gövde\n(β, θ — kamera çerçevesi)", C_GEO, fs=8.7)
box(20, 32.4, 13, 4.2, "Drape (pozlu gövdeye)\n+ clearance + patlama kapısı", C_GEO, fs=8.7)
box(5, 26.0, 28, 4.4,
    "GERÇEK perspektif kamera (pred_cam → K,R,T)\nDiğer açılar: orbit_camera(pelvis, 90/180/270)\nAZİMUTH TAHMİNİ YOK (v1'in ön/arka bug'ı ölü)",
    C_GEO, fs=8.7)
box(5, 19.6, 28, 4.4,
    "pyrender ekran-uzayı pass'leri:\nkamera-uzayı NORMAL + DEPTH + giysi SİLÜETİ\n(kontrolde RGB YOK — v1 halüsinasyon dersi)",
    C_GEO, fs=8.7)
box(5, 13.8, 28, 3.6, "FLUX VAE → kontrol latent'leri (2×64ch)", C_UTIL, fs=9)
arrow(11.5, 43.8, 11.5, 42.6); arrow(26.5, 43.8, 26.5, 42.6)
arrow(11.5, 38.0, 11.5, 36.8); arrow(26.5, 38.0, 26.5, 36.8)
arrow(11.5, 32.2, 13, 30.6); arrow(26.5, 32.2, 25, 30.6)
arrow(19, 25.8, 19, 24.2)
arrow(19, 19.4, 19, 17.6)
route([(5, 15.6), (3.4, 15.6), (3.4, 52.6), (12, 52.6), (12, 53.9)])  # kontrol → hedef token

# --- Kol B: kişi --- #
box(43, 44.0, 18, 3.6, "Kişi Fotoğrafı", C_IO, fs=9.5)
box(41, 37.6, 22, 4.6, "DWPose + ATR parser\n(gerçek onnx backend'ler)", C_UTIL, fs=9)
box(41, 31.2, 22, 4.6, "Agnostic görüntü\n+ inpaint maskesi", C_UTIL, fs=9)
box(41, 24.4, 22, 5.0, "Fill girdileri:\nmasked latent (VAE, 64ch)\nmask 8×8 blok paketi (256ch)", C_UTIL, fs=8.7)
box(41, 17.0, 22, 5.6,
    "EĞİTİM VERİSİ (Aşama 1):\n%70 VITON-HD (gerçek)\n%30 sentetik multi-view\n(0/90/180/270 — GT'li ARKA görünüm)",
    C_TRAIN, fs=8.7)
arrow(52, 43.8, 52, 42.4)
arrow(52, 37.4, 52, 36.0)
arrow(52, 31.0, 52, 29.6)
route([(63, 26.9), (64.8, 26.9), (64.8, 52.6), (54, 52.6), (54, 53.9)])  # → hedef token

# --- Kol C: görünüm --- #
box(72, 44.0, 22, 3.6, "3D Giysi Mesh (.obj + texture)", C_IO, fs=9)
box(70, 37.6, 26, 4.6, "Flat-lit dokulu UV render\n(ürün fotoğrafı benzeri; gölgesiz)", C_APP, fs=9)
box(70, 31.2, 26, 4.6, "Gerçek veride ikame:\nGİYİLEN giysiden parse kesimi\n(süpervizyon tutarlılığı)", C_APP, fs=8.7)
box(70, 24.4, 26, 4.6, "FLUX VAE → referans latent (64ch)\nimg_ids frame_idx = 1 (Kontext tarzı)", C_APP, fs=8.7)
arrow(83, 43.8, 83, 42.4)
arrow(83, 37.4, 83, 36.0)
arrow(83, 31.0, 83, 29.2)
route([(96, 26.7), (97.4, 26.7), (97.4, 52.6), (86, 52.6), (86, 53.9)])  # → referans token

# parite notu (kolonların ALTINDA, ayrı bant)
box(8, 4.6, 84, 5.4,
    "PARİTE SÖZLEŞMESİ: eğitim ön-işleme, sentetik üretici ve inference İSTİSNASIZ aynı build_conditioning()'i çağırır (testle zorlanır)\n"
    "KAPILAR: kamera reproj. IoU ≥ 0.70  •  drape derinlik/patlama reddi  •  Faz 4: kontrol AÇIK ≥ KAPALI ablation  •  eğitilen: ~200M / 12B (%1.7)",
    "#fef7e0", fs=9.3)

fig.savefig("v2/docs/architecture_v2.png", dpi=160, bbox_inches="tight", facecolor="white")
print("yazıldı: v2/docs/architecture_v2.png")
