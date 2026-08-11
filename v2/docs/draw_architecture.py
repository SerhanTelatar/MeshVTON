#!/usr/bin/env python3
"""MeshVTON v2 architecture diagram — code-truthful generation.

Sources: meshvton2/model/flux_tryon.py (channel layout, LoRA, sampler),
model/control_embedder.py (zero-init), conditioning/* (camera, drape, render).
Run: python v2/docs/draw_architecture.py  ->  v2/docs/architecture_v2.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# color palette (related to the v1 diagram)
C_FROZEN = "#dbe9f8"   # frozen backbone (blue)
C_TRAIN = "#d9ead3"    # trained (green)
C_GEO = "#f4cccc"      # 3D/geometry (reddish)
C_APP = "#e6d5f0"      # appearance (purple)
C_UTIL = "#eeeeee"     # helper (grey)
C_IO = "#dce6f5"       # input/output (light blue)
C_TOKEN = "#fff2cc"    # token strip (yellow)

fig, ax = plt.subplots(figsize=(17, 17.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 104)
ax.axis("off")


def box(x, y, w, h, text, fc, fs=10.5, weight="normal", ec="#666666", lw=1.2, style="round,pad=0.35"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)
    return (x + w / 2, y, y + h)  # cx, bottom, top


def arrow(x0, y0, x1, y1, color="#333333", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))


def section(x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc="none",
                                ec="#999999", lw=1.1, linestyle="--"))
    ax.text(x + 0.8, y + h - 1.6, label, fontsize=11, weight="bold", color="#555555", ha="left")


def route(points, color="#2e5f8a", lw=2.0):
    """L-shaped connection routed through the edge corridor without hitting boxes (the last leg carries the arrow)."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="round", zorder=1)
    arrow(points[-2][0], points[-2][1], points[-1][0], points[-1][1], color=color, lw=lw)


ax.text(50, 102.3, "MeshVTON v2: FLUX.1 Fill + Screen-Space Geometry + Reference Tokens",
        ha="center", fontsize=19, weight="bold")

# ------------------------------- TOP: output ------------------------------- #
box(38, 97.2, 24, 2.9, "Try-On Result (0° / 90° / 180° / 270°)", C_TRAIN, weight="bold")
box(38, 93.2, 24, 2.7, "Composite outside the mask (person/background preserved)", C_UTIL, fs=9.5)
box(38, 89.3, 24, 2.7, "FLUX VAE Decoder", C_FROZEN)
arrow(50, 92.2, 50, 93.0)
arrow(50, 96.1, 50, 97.0)

# --------------------------- MIDDLE: FLUX backbone --------------------------- #
section(8, 60.5, 84, 26.5, "BACKBONE — FLUX.1 Fill-dev Transformer (12B DiT, FROZEN)")
box(11, 78.5, 37, 5.6,
    "ControlXEmbedder\nx_embedder (384ch, frozen)  +  control_proj (128ch)\nZERO-INIT → bit-identical stock Fill at the start",
    C_TRAIN, fs=9.5)
box(52, 78.5, 37, 5.6,
    "LoRA  r=64  (attn q/k/v/out + FF)\n~200M trainable parameters\n(task: dress from the reference + read the geometry)",
    C_TRAIN, fs=9.5)
box(11, 70.5, 78, 5.2,
    "57 blocks — full attention: target ↔ reference ↔ text tokens\n"
    "T5/CLIP text embeddings computed ONCE for the fixed prompt (the encoders are not kept in memory)",
    C_FROZEN, fs=9.5)
box(11, 63.0, 78, 5.0,
    "Rectified Flow — training: logit-normal t + resolution shift, loss ONLY on the target tokens\n"
    "inference: Euler, 28 steps (FluxTryOnSampler mirrors the training forward)",
    C_FROZEN, fs=9.5)
arrow(50, 86.9, 50, 89.1)  # backbone -> decoder direction (upward)

# --------------------------- token sequence strip --------------------------- #
box(8, 54.0, 50, 4.6,
    "TARGET tokens (frame 0):\n[ x_t 64 | masked 64 | mask 256 | control 128 ]  = 512 channels",
    C_TOKEN, fs=9.5)
box(61, 54.0, 31, 4.6,
    "REFERENCE tokens (frame 1):\n[ ref 64 | ref 64 | 0·256 | 0·128 ]",
    C_TOKEN, fs=9.5)
arrow(33, 58.6, 40, 60.7)
arrow(76.5, 58.6, 65, 60.7)

# ------------------------------ BOTTOM: 3 branches ------------------------------ #
section(2, 11.8, 34, 39.2, "GEOMETRY (screen-space)")
section(38, 11.8, 28, 39.2, "PERSON")
section(68, 11.8, 30, 39.2, "APPEARANCE")

# --- Branch A: geometry --- #
box(5, 44.0, 13, 3.6, "Person Photo", C_IO, fs=9.5)
box(20, 44.0, 13, 3.6, "3D Garment Mesh\n(.obj + texture)", C_IO, fs=9)
box(5, 38.2, 13, 4.2, "HMR2 (4D-Humans)\npred_cam + bbox\nNO LONGER DISCARDED", C_GEO, fs=8.7)
box(20, 38.2, 13, 4.2, "LBS surface binding\n(once at rest, .npz cache)", C_GEO, fs=8.7)
box(5, 32.4, 13, 4.2, "SMPL-X body\n(β, θ — camera frame)", C_GEO, fs=8.7)
box(20, 32.4, 13, 4.2, "Drape (onto the posed body)\n+ clearance + explosion gate", C_GEO, fs=8.7)
box(5, 26.0, 28, 4.4,
    "REAL perspective camera (pred_cam → K,R,T)\nOther angles: orbit_camera(pelvis, 90/180/270)\nNO AZIMUTH ESTIMATION (v1's front/back bug is dead)",
    C_GEO, fs=8.7)
box(5, 19.6, 28, 4.4,
    "pyrender screen-space passes:\ncamera-space NORMAL + DEPTH + garment SILHOUETTE\n(NO RGB in the control — the v1 hallucination lesson)",
    C_GEO, fs=8.7)
box(5, 13.8, 28, 3.6, "FLUX VAE → control latents (2×64ch)", C_UTIL, fs=9)
arrow(11.5, 43.8, 11.5, 42.6); arrow(26.5, 43.8, 26.5, 42.6)
arrow(11.5, 38.0, 11.5, 36.8); arrow(26.5, 38.0, 26.5, 36.8)
arrow(11.5, 32.2, 13, 30.6); arrow(26.5, 32.2, 25, 30.6)
arrow(19, 25.8, 19, 24.2)
arrow(19, 19.4, 19, 17.6)
route([(5, 15.6), (3.4, 15.6), (3.4, 52.6), (12, 52.6), (12, 53.9)])  # control → target token

# --- Branch B: person --- #
box(43, 44.0, 18, 3.6, "Person Photo", C_IO, fs=9.5)
box(41, 37.6, 22, 4.6, "DWPose + ATR parser\n(real onnx backends)", C_UTIL, fs=9)
box(41, 31.2, 22, 4.6, "Agnostic image\n+ inpaint mask", C_UTIL, fs=9)
box(41, 24.4, 22, 5.0, "Fill inputs:\nmasked latent (VAE, 64ch)\nmask 8×8 block pack (256ch)", C_UTIL, fs=8.7)
box(41, 17.0, 22, 5.6,
    "TRAINING DATA (Stage 1):\n70% VITON-HD (real)\n30% synthetic multi-view\n(0/90/180/270 — BACK view with GT)",
    C_TRAIN, fs=8.7)
arrow(52, 43.8, 52, 42.4)
arrow(52, 37.4, 52, 36.0)
arrow(52, 31.0, 52, 29.6)
route([(63, 26.9), (64.8, 26.9), (64.8, 52.6), (54, 52.6), (54, 53.9)])  # → target token

# --- Branch C: appearance --- #
box(72, 44.0, 22, 3.6, "3D Garment Mesh (.obj + texture)", C_IO, fs=9)
box(70, 37.6, 26, 4.6, "Flat-lit textured UV render\n(product-photo-like; unshaded)", C_APP, fs=9)
box(70, 31.2, 26, 4.6, "Substitute on real data:\nparse crop of the WORN garment\n(supervision consistency)", C_APP, fs=8.7)
box(70, 24.4, 26, 4.6, "FLUX VAE → reference latent (64ch)\nimg_ids frame_idx = 1 (Kontext style)", C_APP, fs=8.7)
arrow(83, 43.8, 83, 42.4)
arrow(83, 37.4, 83, 36.0)
arrow(83, 31.0, 83, 29.2)
route([(96, 26.7), (97.4, 26.7), (97.4, 52.6), (86, 52.6), (86, 53.9)])  # → reference token

# parity note (BELOW the columns, separate band)
box(8, 4.6, 84, 5.4,
    "PARITY CONTRACT: training preprocessing, the synthetic generator and inference all call the same build_conditioning() WITHOUT EXCEPTION (enforced by tests)\n"
    "GATES: camera reproj. IoU ≥ 0.70  •  drape depth/explosion rejection  •  Phase 4: control ON ≥ OFF ablation  •  trained: ~200M / 12B (1.7%)",
    "#fef7e0", fs=9.3)

fig.savefig("v2/docs/architecture_v2.png", dpi=160, bbox_inches="tight", facecolor="white")
print("written: v2/docs/architecture_v2.png")
