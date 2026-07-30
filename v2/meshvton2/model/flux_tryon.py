"""FLUX try-on sarmalayıcısı — TÜM diffusers temas noktaları bu dosyada.

Faz 1 (zero-shot, eğitimsiz) varyantları:
- "fill_spatial": FluxFillPipeline; canvas = [görünüm referansı | agnostic kişi],
  maske yalnız kişi yarısında → model referansı görerek inpaint eder (CatVTON hilesi).
- "kontext": FluxKontextPipeline; giriş = [kişi | referans] dikişli tek görüntü +
  edit talimatı, çıktı sol yarıdan kırpılır.

Plan notu: üçüncü varyant (Fill + eğitimsiz Kontext-tarzı ref-token sequence-concat)
Faz 4'e ertelendi — eğitimsiz halinin kanıt değeri düşük (Fill ref-token görmeden
eğitildi, OOD) ve özel örnekleme döngüsü zaten Faz 4'te reference_tokens.py ile
geliyor. Karar Faz 1 raporunda (a) vs (c) kanıtıyla verilir.

Maske disiplini: fill_spatial çıktısında maske DIŞI pikseller orijinal kişiyle
kompozit edilir — zero-shot modelin kişi/arka planı bozması metriklere sızmaz.
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


# ------------------------- saf yardımcılar (lokal testlenebilir) ------------------------- #


def make_side_canvas(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """İki (H,W,3) görüntüyü yatay dikişle (H,2W,3) yapar; boyutlar eşit olmalı."""
    if left.shape != right.shape:
        raise ValueError(f"Boyut uyuşmazlığı: {left.shape} vs {right.shape}")
    return np.concatenate([left, right], axis=1)


def make_side_mask(mask_right: np.ndarray) -> np.ndarray:
    """(H,W) maskeyi (H,2W) canvas maskesine koyar: sol yarı (referans) daima 0."""
    h, w = mask_right.shape[:2]
    out = np.zeros((h, 2 * w), dtype=np.uint8)
    out[:, w:] = mask_right
    return out


def crop_half(canvas: np.ndarray, side: str) -> np.ndarray:
    h, w2 = canvas.shape[:2]
    w = w2 // 2
    return canvas[:, :w] if side == "left" else canvas[:, w:]


def composite_outside_mask(pred: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Maske dışını orijinalden geri koy (kenarda 3px yumuşatma)."""
    import cv2

    m = ((mask > 127) * 255).astype(np.uint8)
    m = cv2.GaussianBlur(m, (7, 7), 0).astype(np.float32) / 255.0
    m = m[..., None]
    out = pred.astype(np.float32) * m + original.astype(np.float32) * (1 - m)
    return out.round().astype(np.uint8)


# ------------------------------- pipeline sarmalayıcı ------------------------------- #


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
            raise ValueError(f"variant {VARIANTS} içinden olmalı, gelen: {variant}")
        self.variant = variant
        if dtype is None:  # T4 gibi Turing GPU'lar bf16'yı desteklemez → fp16
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
        """Tek try-on üretimi; (H,W,3) uint8 RGB döner."""
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
                guidance_scale=30.0,  # Fill-dev önerilen yüksek guidance
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
            if arr.shape[:2] != (h, 2 * w):  # Kontext tercih ettiği çözünürlüğe kayabilir
                arr = np.asarray(Image.fromarray(arr).resize((2 * w, h), Image.LANCZOS))
            pred = crop_half(arr, "left")

        return composite_outside_mask(pred, person, mask)



# --------------------------------------------------------------------------- #
# Faz 4: eğitim — FluxTrainModule (FLUX'a dokunan TEK yer burası kalmaya devam)
# --------------------------------------------------------------------------- #

from meshvton2.model.reference_tokens import (  # noqa: E402
    concat_reference,
    make_img_ids,
    pack_latents,
    pack_pixel_mask,
)

FILL_BASE_CH = 384  # 64 (noisy latent) + 64 (masked image latent) + 256 (mask blokları)


def mask_image_for_fill(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill'in beklediği maskeli görüntü: maske içi SİYAH.

    Dikkat: [-1,1] uzayında img*(1-mask) siyah değil GRİ (0) üretir — iki taslak
    sürümde de bu hata vardı. Doğrusu [0,1]'e çık, maskele, geri dön.
    """
    img01 = (img + 1.0) / 2.0
    return (img01 * (1.0 - mask)) * 2.0 - 1.0


def assemble_train_sequence(
    xt_lat: torch.Tensor,          # (B,16,h,w) gürültülü hedef latent
    masked_lat: torch.Tensor,      # (B,16,h,w) maskeli görüntü latent'i
    pixel_mask: torch.Tensor,      # (B,1,8h,8w) inpaint maskesi {0,1}
    control_lats: list[torch.Tensor],  # her biri (B,16,h,w) — [normal, depth_sil]
    ref_lat: torch.Tensor,         # (B,16,h,w) görünüm referansı latent'i
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Fill kanal düzeni + Kontext-tarzı referans concat (saf tensör — lokal testli).

    hedef token = [x_t 64 | masked 64 | mask 256 | kontrol 64×N] = 384+64N
    ref   token = [ref 64 | ref 64    | 0    256 | 0      64×N]
      (ref'in "masked image" yuvası kendisi: temiz bağlam; mask=0 → inpaint edilmez)

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
    """FLUX.1 Fill + LoRA + zero-init kontrol embedder'ı eğitim sarmalayıcısı.

    Dondurulmuş Fill transformer + VAE; eğitilen = LoRA + control_proj.
    Sabit prompt'un T5/CLIP embed'leri setup'ta CPU'da BİR KEZ hesaplanır
    (GPU'da ~9GB T5 tepe kullanımı olmaz), text encoder'lar atılır.
    step(batch) -> loss; TrainLoop'a step_fn olarak verilir. Eğitilebilir
    durum = transformer'daki requires_grad parametreler (tek checkpoint).
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
        train_guidance: float = 1.0,   # Fill guidance-distilled; eğitimde sabit
        ref_dropout: float = 0.1,      # referansı %10 gri yap → geometri yük taşımayı öğrenir
        t_mean: float = 0.0,
        t_std: float = 1.0,
        compile_transformer: bool = False,  # torch.compile (~1.3x; ilk adımda dakikalarca derleme)
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

    # ------------------------------ kurulum ------------------------------ #

    def setup(self):
        import gc

        from diffusers import FluxFillPipeline

        from meshvton2.model.control_embedder import attach_control_embedder
        from meshvton2.model.lora import attach_lora, count_trainable

        dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float16)
        self.dtype = dtype
        pipe = FluxFillPipeline.from_pretrained(self.repo, torch_dtype=dtype)

        # Sabit prompt embed'leri CPU'da bir kez → text encoder'ları at
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
        # ckpt/durum isimleri değişmesin diye ham modül referansı (compile '_orig_mod.' öneki ekler)
        self._raw_transformer = self.transformer
        if self.compile_transformer:
            try:
                self.transformer = torch.compile(self.transformer, dynamic=False)
                print("torch.compile aktif — İLK adım dakikalarca sürebilir (derleme), sonrası hızlı")
            except Exception as e:  # compile başarısızlığı eğitimi durdurmasın
                print(f"UYARI: torch.compile başarısız ({e}) — derlenmemiş devam")

        del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2, pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"FluxTrainModule hazır — eğitilebilir param: {count_trainable(self.transformer):,}")
        return self

    # ------------------------------ yardımcılar ------------------------------ #

    @torch.no_grad()
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) [-1,1] -> FLUX latent (B,16,H/8,W/8), ölçekli."""
        img = img.to(self.device, self.vae.dtype)
        z = self.vae.encode(img).latent_dist.sample()
        return (z - self.vae_shift) * self.vae_scale

    def _gray_latent(self, hw: tuple[int, int]) -> torch.Tensor:
        """ref_dropout için nötr gri görüntünün latent'i (şekil başına bir kez)."""
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

        # HAM modül üzerinden: compile sarmalayıcısı isimlere '_orig_mod.' öneki
        # ekler — ckpt isim uzayı derleme durumundan bağımsız kalmalı (resume uyumu)
        return trainable_state(getattr(self, "_raw_transformer", self.transformer))

    def load_trainable_state(self, state: dict) -> None:
        from meshvton2.model.lora import load_trainable_state

        load_trainable_state(getattr(self, "_raw_transformer", self.transformer), state)

    # -------------------------------- adım -------------------------------- #

    def step(self, batch: dict) -> torch.Tensor:
        from meshvton2.training.flow_matching import (
            apply_shift,
            resolution_shift,
            rf_interpolate,
            rf_target,
            sample_logit_normal_t,
        )

        mask_cpu = batch["inpaint_mask"]
        if "gt_lat" in batch:  # precompute_latents.py yolu: VAE yok, PNG çözme yok (saf hız)
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
                    ref[drop] = 0.0  # [-1,1] ortası = nötr gri görüntü
            ref_lat = self.encode(ref)

        b = x0.shape[0]
        noise = torch.randn(x0.shape, generator=self.gen).to(self.device, torch.float32)
        t = sample_logit_normal_t(b, mean=self.t_mean, std=self.t_std, generator=self.gen).to(self.device)
        t = apply_shift(t, resolution_shift((x0.shape[2] // 2) * (x0.shape[3] // 2)))
        x_t = rf_interpolate(x0.float(), noise, t).to(x0.dtype)  # fp32 karışım, sonra geri

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
    """Eğitilmiş checkpoint'le try-on üretimi — FluxTrainModule.step'in AYNASI.

    Aynı kanal düzeni, aynı referans token'ları, aynı guidance (eğitimdekiyle
    tutarlı olmalı: LoRA guidance=1.0 altında adapte edildi). Euler örnekleme:
    x_{i+1} = x_i + (σ_{i+1} − σ_i)·v.

    control_scale: 1.0 = eğitilmiş kontrol; 0.0 = kontrol latent'leri sıfır →
    control_proj(0)=0, bit-eş stok davranış — '--disable-control' ABLATION
    KAPISI bununla ölçülür (v1 FAZ C dersinin otomatik testi).
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
            state = ck.get("trainables") or ck  # TrainLoop ckpt'i veya düz sözlük
            self.module.load_trainable_state(state)
            print(f"checkpoint yüklendi: {checkpoint} ({len(state)} tensör)")
        self.module.transformer.eval()

    @torch.no_grad()
    def sample(self, bundle, *, steps: int = 28, seed: int = 0,
               control_scale: float = 1.0, guidance: float | None = None,
               return_raw: bool = False):
        """bundle: ConditioningBundle (veya aynı alanlara sahip dict-benzeri).
        guidance: None -> eğitimdeki değer (1.0); FLUX Fill guidance-damıtılmış,
        yüksek guidance (3.5-30) sadakat/doygunluğu artırır (solgunluk çaresi).
        return_raw: True ise (composited, raw_pred) döner — raw_pred kompozit
        ÖNCESİ VAE çıktısıdır (maske dışı geri-yapıştırma yok); saydamlık/leke
        modelin üretiminden mi yoksa kompozit kenar yumuşatmasından mı geldiğini
        ayırt etmek için (bkz. [[meshvton-v2-inference-alignment]] teşhis akışı).
        -> (H,W,3) uint8 try-on; maske dışı agnostic'ten kompozit edilir."""
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
