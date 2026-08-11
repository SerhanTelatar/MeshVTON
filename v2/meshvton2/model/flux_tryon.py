"""FLUX try-on wrapper — ALL diffusers touch points live in this file.

Phase 1 (zero-shot, untrained) variants:
- "fill_spatial": FluxFillPipeline; canvas = [appearance reference | agnostic person],
  the mask covers only the person half → the model inpaints while seeing the reference (the CatVTON trick).
- "kontext": FluxKontextPipeline; input = a single stitched image [person | reference] +
  an edit instruction, the output is cropped from the left half.

Plan note: the third variant (Fill + untrained Kontext-style ref-token sequence concat)
was deferred to Phase 4 — its untrained form has little evidential value (Fill was trained
without ever seeing ref tokens, so it is OOD) and the custom sampling loop already arrives
in Phase 4 via reference_tokens.py. The decision is made from the (a) vs (c) evidence in the Phase 1 report.

Mask discipline: in the fill_spatial output, pixels OUTSIDE the mask are composited back
from the original person — a zero-shot model damaging the person/background cannot leak into the metrics.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

VARIANTS = ("fill_spatial", "kontext")

FILL_PROMPT = (
    "the person is wearing the garment shown on the left, "
    "photorealistic fashion photo, natural fit, correct drape"
)
KONTEXT_PROMPT = (
    "Make the person on the left wear the garment shown on the right. "
    "Keep the person's identity, pose, body shape and background unchanged."
)


# ------------------------- pure helpers (locally testable) ------------------------- #


def make_side_canvas(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stitches two (H,W,3) images horizontally into (H,2W,3); the sizes must match."""
    if left.shape != right.shape:
        raise ValueError(f"Size mismatch: {left.shape} vs {right.shape}")
    return np.concatenate([left, right], axis=1)


def make_side_mask(mask_right: np.ndarray) -> np.ndarray:
    """Places an (H,W) mask into an (H,2W) canvas mask: the left half (reference) is always 0."""
    h, w = mask_right.shape[:2]
    out = np.zeros((h, 2 * w), dtype=np.uint8)
    out[:, w:] = mask_right
    return out


def crop_half(canvas: np.ndarray, side: str) -> np.ndarray:
    h, w2 = canvas.shape[:2]
    w = w2 // 2
    return canvas[:, :w] if side == "left" else canvas[:, w:]


def composite_outside_mask(pred: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Restore everything outside the mask from the original (3px feather at the edge)."""
    import cv2

    m = ((mask > 127) * 255).astype(np.uint8)
    m = cv2.GaussianBlur(m, (7, 7), 0).astype(np.float32) / 255.0
    m = m[..., None]
    out = pred.astype(np.float32) * m + original.astype(np.float32) * (1 - m)
    return out.round().astype(np.uint8)


# ------------------------------- pipeline wrapper ------------------------------- #


class FluxTryOn:
    def __init__(
        self,
        variant: str,
        *,
        fill_repo: str = "black-forest-labs/FLUX.1-Fill-dev",
        kontext_repo: str = "black-forest-labs/FLUX.1-Kontext-dev",
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        steps: int = 28,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got: {variant}")
        self.variant = variant
        if dtype is None:  # Turing GPUs such as the T4 do not support bf16 → fp16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.device, self.dtype, self.steps = device, dtype, steps
        self.repo = fill_repo if variant == "fill_spatial" else kontext_repo
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        if self.variant == "fill_spatial":
            from diffusers import FluxFillPipeline

            self._pipe = FluxFillPipeline.from_pretrained(self.repo, torch_dtype=self.dtype)
        else:
            from diffusers import FluxKontextPipeline

            self._pipe = FluxKontextPipeline.from_pretrained(self.repo, torch_dtype=self.dtype)
        self._pipe.enable_model_cpu_offload() if self.device == "cuda_offload" else self._pipe.to(self.device)

    @torch.no_grad()
    def tryon(
        self,
        person: np.ndarray,       # (H,W,3) uint8 RGB
        agnostic: np.ndarray,     # (H,W,3) uint8 RGB
        mask: np.ndarray,         # (H,W) uint8 {0,255}
        appearance_ref: np.ndarray,  # (H,W,3) uint8 RGB
        *,
        seed: int = 0,
        prompt: str | None = None,
    ) -> np.ndarray:
        """A single try-on generation; returns (H,W,3) uint8 RGB."""
        self._load()
        h, w = person.shape[:2]
        gen = torch.Generator(device="cpu").manual_seed(seed)

        if self.variant == "fill_spatial":
            canvas = make_side_canvas(appearance_ref, agnostic)
            cmask = make_side_mask(mask)
            out = self._pipe(
                prompt=prompt or FILL_PROMPT,
                image=Image.fromarray(canvas),
                mask_image=Image.fromarray(cmask),
                height=h,
                width=2 * w,
                num_inference_steps=self.steps,
                guidance_scale=30.0,  # high guidance recommended for Fill-dev
                generator=gen,
            ).images[0]
            pred = crop_half(np.asarray(out.convert("RGB")), "right")
        else:
            stitched = make_side_canvas(person, appearance_ref)
            out = self._pipe(
                prompt=prompt or KONTEXT_PROMPT,
                image=Image.fromarray(stitched),
                height=h,
                width=2 * w,
                num_inference_steps=self.steps,
                guidance_scale=2.5,
                generator=gen,
            ).images[0]
            arr = np.asarray(out.convert("RGB"))
            if arr.shape[:2] != (h, 2 * w):  # Kontext may drift to its preferred resolution
                arr = np.asarray(Image.fromarray(arr).resize((2 * w, h), Image.LANCZOS))
            pred = crop_half(arr, "left")

        return composite_outside_mask(pred, person, mask)



# --------------------------------------------------------------------------- #
# Phase 4: training — FluxTrainModule (still the ONLY place that touches FLUX)
# --------------------------------------------------------------------------- #

from meshvton2.model.reference_tokens import (  # noqa: E402
    concat_reference,
    make_img_ids,
    pack_latents,
    pack_pixel_mask,
)

FILL_BASE_CH = 384  # 64 (noisy latent) + 64 (masked image latent) + 256 (mask blocks)


def mask_image_for_fill(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The masked image Fill expects: BLACK inside the mask.

    Careful: in [-1,1] space img*(1-mask) produces GREY (0), not black — both draft
    versions had this bug. The correct way is to go to [0,1], mask, and come back.
    """
    img01 = (img + 1.0) / 2.0
    return (img01 * (1.0 - mask)) * 2.0 - 1.0


def assemble_train_sequence(
    xt_lat: torch.Tensor,          # (B,16,h,w) noisy target latent
    masked_lat: torch.Tensor,      # (B,16,h,w) masked image latent
    pixel_mask: torch.Tensor,      # (B,1,8h,8w) inpaint mask {0,1}
    control_lats: list[torch.Tensor],  # each (B,16,h,w) — [normal, depth_sil]
    ref_lat: torch.Tensor,         # (B,16,h,w) appearance reference latent
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Fill channel layout + Kontext-style reference concat (pure tensors — locally tested).

    target token = [x_t 64 | masked 64 | mask 256 | control 64×N] = 384+64N
    ref    token = [ref 64 | ref 64    | 0    256 | 0       64×N]
      (the ref's "masked image" slot is itself: clean context; mask=0 → it is not inpainted)

    Returns: (tokens (B,Lt+Lr,384+64N), img_ids (Lt+Lr,3), Lt)
    """
    b, _, h, w = xt_lat.shape
    xt_p, masked_p, ref_p = pack_latents(xt_lat), pack_latents(masked_lat), pack_latents(ref_lat)
    mask_p = pack_pixel_mask(pixel_mask, h, w)
    ctrl_p = [pack_latents(c) for c in control_lats]

    target = torch.cat([xt_p, masked_p, mask_p, *ctrl_p], dim=-1)
    reference = torch.cat(
        [ref_p, ref_p, torch.zeros_like(mask_p), *[torch.zeros_like(c) for c in ctrl_p]], dim=-1
    )
    tids = make_img_ids(h, w, frame_idx=0, device=xt_lat.device)
    rids = make_img_ids(h, w, frame_idx=1, device=xt_lat.device)
    tokens, ids, _ = concat_reference(target, tids, reference, rids)
    return tokens, ids, target.shape[1]


class FluxTrainModule:
    """FLUX.1 Fill + LoRA + zero-init control embedder training wrapper.

    Frozen Fill transformer + VAE; trained = LoRA + control_proj.
    The fixed prompt's T5/CLIP embeddings are computed ONCE on the CPU during setup
    (no ~9GB T5 peak on the GPU), then the text encoders are dropped.
    step(batch) -> loss; passed to TrainLoop as step_fn. The trainable state
    = the requires_grad parameters in the transformer (a single checkpoint).
    """

    def __init__(
        self,
        repo: str = "black-forest-labs/FLUX.1-Fill-dev",
        *,
        prompt: str = "a person wearing the garment, photorealistic fashion photo",
        device: str = "cuda",
        lora_rank: int = 64,
        lora_alpha: int = 64,
        control_images: int = 2,       # normal + depth_sil
        train_guidance: float = 1.0,   # Fill is guidance-distilled; fixed during training
        ref_dropout: float = 0.1,      # grey out the reference 10% of the time → it learns to carry load from geometry
        t_mean: float = 0.0,
        t_std: float = 1.0,
        compile_transformer: bool = False,  # torch.compile (~1.3x; minutes of compilation on the first step)
        seed: int = 0,
    ):
        self.compile_transformer = compile_transformer
        self.repo, self.prompt, self.device = repo, prompt, device
        self.lora_rank, self.lora_alpha = lora_rank, lora_alpha
        self.control_in = 64 * control_images
        self.train_guidance, self.ref_dropout = train_guidance, ref_dropout
        self.t_mean, self.t_std = t_mean, t_std
        self.gen = torch.Generator().manual_seed(seed)
        self.transformer = None

    # ------------------------------ setup ------------------------------ #

    def setup(self):
        import gc

        from diffusers import FluxFillPipeline

        from meshvton2.model.control_embedder import attach_control_embedder
        from meshvton2.model.lora import attach_lora, count_trainable

        dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float16)
        self.dtype = dtype
        pipe = FluxFillPipeline.from_pretrained(self.repo, torch_dtype=dtype)

        # Fixed prompt embeddings once on the CPU → drop the text encoders
        with torch.no_grad():
            pe, ppe, tids = pipe.encode_prompt(
                prompt=self.prompt, prompt_2=None, device="cpu", num_images_per_prompt=1
            )
        self.prompt_embeds = pe.to(self.device, dtype)   # (1, 512, 4096)
        self.pooled_embeds = ppe.to(self.device, dtype)  # (1, 768)
        self.text_ids = tids.to(self.device)             # (512, 3)

        self.vae = pipe.vae.to(self.device).requires_grad_(False).eval()
        self.vae_scale = self.vae.config.scaling_factor
        self.vae_shift = self.vae.config.shift_factor

        self.transformer = pipe.transformer.to(self.device)
        self.transformer.requires_grad_(False)
        attach_lora(self.transformer, rank=self.lora_rank, alpha=self.lora_alpha)
        self.control_embedder = attach_control_embedder(self.transformer, self.control_in)
        self.transformer.enable_gradient_checkpointing()
        self.transformer.train()
        # Raw module reference so ckpt/state names stay stable (compile adds a '_orig_mod.' prefix)
        self._raw_transformer = self.transformer
        if self.compile_transformer:
            try:
                self.transformer = torch.compile(self.transformer, dynamic=False)
                print("torch.compile enabled — the FIRST step may take minutes (compilation), then it is fast")
            except Exception as e:  # a compile failure must not stop training
                print(f"WARNING: torch.compile failed ({e}) — continuing uncompiled")

        del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2, pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"FluxTrainModule ready — trainable params: {count_trainable(self.transformer):,}")
        return self

    # ------------------------------ helpers ------------------------------ #

    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) [-1,1] -> FLUX latent (B,16,H/8,W/8), scaled."""
        img = img.to(self.device, self.vae.dtype)
        z = self.vae.encode(img).latent_dist.sample()
        return (z - self.vae_shift) * self.vae_scale

    def _gray_latent(self, hw: tuple[int, int]) -> torch.Tensor:
        """Latent of the neutral grey image used for ref_dropout (computed once per shape)."""
        key = tuple(hw)
        cache = getattr(self, "_gray_cache", None) or {}
        if key not in cache:
            gray = torch.zeros(1, 3, hw[0] * 8, hw[1] * 8)
            cache[key] = self.encode(gray)[0]
            self._gray_cache = cache
        return cache[key]

    def trainable_parameters(self):
        return (p for p in self.transformer.parameters() if p.requires_grad)

    def trainable_state(self) -> dict:
        from meshvton2.model.lora import trainable_state

        # Via the RAW module: the compile wrapper adds a '_orig_mod.' prefix to names —
        # the ckpt namespace must stay independent of compile state (resume compatibility)
        return trainable_state(getattr(self, "_raw_transformer", self.transformer))

    def load_trainable_state(self, state: dict) -> None:
        from meshvton2.model.lora import load_trainable_state

        load_trainable_state(getattr(self, "_raw_transformer", self.transformer), state)

    # -------------------------------- step -------------------------------- #

    def step(self, batch: dict) -> torch.Tensor:
        from meshvton2.training.flow_matching import (
            apply_shift,
            resolution_shift,
            rf_interpolate,
            rf_target,
            sample_logit_normal_t,
        )

        mask_cpu = batch["inpaint_mask"]
        if "gt_lat" in batch:  # precompute_latents.py path: no VAE, no PNG decode (pure speed)
            to_dev = lambda k: batch[k].to(self.device, self.vae.dtype)
            x0 = to_dev("gt_lat")
            masked_lat = to_dev("masked_lat")
            ctrl_lats = [to_dev("normal_lat"), to_dev("depth_sil_lat")]
            ref_lat = to_dev("ref_lat")
            if self.ref_dropout > 0:
                drop = torch.rand(ref_lat.shape[0], generator=self.gen) < self.ref_dropout
                if drop.any():
                    ref_lat = ref_lat.clone()
                    ref_lat[drop] = self._gray_latent(ref_lat.shape[-2:])
        else:
            x0 = self.encode(batch["gt_rgb"])
            masked_lat = self.encode(mask_image_for_fill(batch["agnostic_rgb"], mask_cpu))
            ctrl_lats = [self.encode(batch["control_normal"]), self.encode(batch["control_depth_sil"])]
            ref = batch["appearance_ref"]
            if self.ref_dropout > 0:
                drop = torch.rand(ref.shape[0], generator=self.gen) < self.ref_dropout
                if drop.any():
                    ref = ref.clone()
                    ref[drop] = 0.0  # the middle of [-1,1] = a neutral grey image
            ref_lat = self.encode(ref)

        b = x0.shape[0]
        noise = torch.randn(x0.shape, generator=self.gen).to(self.device, torch.float32)
        t = sample_logit_normal_t(b, mean=self.t_mean, std=self.t_std, generator=self.gen).to(self.device)
        t = apply_shift(t, resolution_shift((x0.shape[2] // 2) * (x0.shape[3] // 2)))
        x_t = rf_interpolate(x0.float(), noise, t).to(x0.dtype)  # fp32 mix, then back

        mask = mask_cpu.to(self.device, x0.dtype)
        tokens, img_ids, lt = assemble_train_sequence(x_t, masked_lat, mask, ctrl_lats, ref_lat)

        v_pred = self.transformer(
            hidden_states=tokens.to(self.transformer.dtype),
            timestep=t.to(self.transformer.dtype),
            guidance=torch.full((b,), self.train_guidance, device=self.device, dtype=torch.float32),
            pooled_projections=self.pooled_embeds.expand(b, -1),
            encoder_hidden_states=self.prompt_embeds.expand(b, -1, -1),
            txt_ids=self.text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        v_target = pack_latents(rf_target(x0.float(), noise))
        return torch.nn.functional.mse_loss(v_pred[:, :lt].float(), v_target)


class FluxTryOnSampler:
    """Try-on generation with a trained checkpoint — the MIRROR of FluxTrainModule.step.

    Same channel layout, same reference tokens, same guidance (it must match training:
    the LoRA was adapted under guidance=1.0). Euler sampling:
    x_{i+1} = x_i + (σ_{i+1} − σ_i)·v.

    control_scale: 1.0 = trained control; 0.0 = zero control latents →
    control_proj(0)=0, bit-identical stock behaviour — the '--disable-control' ABLATION
    GATE is measured with this (the automated test of the v1 PHASE C lesson).
    """

    def __init__(self, repo: str = "black-forest-labs/FLUX.1-Fill-dev", *,
                 checkpoint: str | None = None, device: str = "cuda",
                 prompt: str = "a person wearing the garment, photorealistic fashion photo",
                 guidance: float = 1.0, control_images: int = 2):
        self.module = FluxTrainModule(
            repo, prompt=prompt, device=device, control_images=control_images,
            train_guidance=guidance, ref_dropout=0.0,
        ).setup()
        if checkpoint:
            import torch as _t

            ck = _t.load(checkpoint, map_location="cpu", weights_only=False)
            state = ck.get("trainables") or ck  # a TrainLoop ckpt or a plain dict
            self.module.load_trainable_state(state)
            print(f"checkpoint loaded: {checkpoint} ({len(state)} tensors)")
        self.module.transformer.eval()

    @torch.no_grad()
    def sample(self, bundle, *, steps: int = 28, seed: int = 0,
               control_scale: float = 1.0, guidance: float | None = None,
               return_raw: bool = False):
        """bundle: ConditioningBundle (or anything dict-like with the same fields).
        guidance: None -> the training value (1.0); FLUX Fill is guidance-distilled,
        high guidance (3.5-30) raises fidelity/saturation (the cure for washed-out output).
        return_raw: if True returns (composited, raw_pred) — raw_pred is the VAE output
        BEFORE compositing (no paste-back outside the mask); it tells apart whether
        transparency/smearing comes from the model's generation or from the composite
        edge feathering (see the [[meshvton-v2-inference-alignment]] diagnostic flow).
        -> (H,W,3) uint8 try-on; outside the mask it is composited from the agnostic."""
        from meshvton2.model.reference_tokens import unpack_latents
        from meshvton2.training.flow_matching import make_sigma_schedule

        m = self.module
        dev, dt = m.device, m.dtype
        unsq = lambda t: t.unsqueeze(0).to(dev)
        agnostic = unsq(bundle.agnostic_rgb)
        mask = unsq(bundle.inpaint_mask)
        _, _, hgt, wdt = agnostic.shape
        lh, lw = hgt // 8, wdt // 8

        masked_lat = m.encode(mask_image_for_fill(agnostic.cpu(), mask.cpu()))
        ctrl_lats = [m.encode(unsq(bundle.control_normal)),
                     m.encode(unsq(bundle.control_depth_sil))]
        if control_scale != 1.0:
            ctrl_lats = [c * control_scale for c in ctrl_lats]
        ref_lat = m.encode(unsq(bundle.appearance_ref))

        g_val = m.train_guidance if guidance is None else guidance
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn((1, 16, lh, lw), generator=gen).to(dev, torch.float32)
        sigmas = make_sigma_schedule(steps, (lh // 2) * (lw // 2)).to(dev)

        for i in range(steps):
            tokens, img_ids, lt = assemble_train_sequence(
                x.to(dt), masked_lat, mask.to(dev, dt), ctrl_lats, ref_lat
            )
            v = m.transformer(
                hidden_states=tokens.to(m.transformer.dtype),
                timestep=sigmas[i].expand(1).to(m.transformer.dtype),
                guidance=torch.full((1,), g_val, device=dev, dtype=torch.float32),
                pooled_projections=m.pooled_embeds,
                encoder_hidden_states=m.prompt_embeds,
                txt_ids=m.text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0][:, :lt]
            v = unpack_latents(v.float(), lh, lw)
            x = x + (sigmas[i + 1] - sigmas[i]) * v

        lat = x.to(dt) / m.vae_scale + m.vae_shift
        img = m.vae.decode(lat).sample[0]  # (3,H,W) [-1,1]
        pred = ((img.float().clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).cpu().numpy()

        person = ((bundle.agnostic_rgb.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()
        mask_u8 = (bundle.inpaint_mask[0].numpy() * 255).astype(np.uint8)
        composited = composite_outside_mask(pred, person, mask_u8)
        return (composited, pred) if return_raw else composited
