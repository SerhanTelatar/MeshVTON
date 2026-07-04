"""Genel eğitim döngüsü (v1 trainer.py iskeletinin sadeleşmiş hali).

Model/backbone bilgisi YOK: `step_fn(batch) -> loss` enjekte edilir (FLUX'a dokunan
her şey model/flux_tryon.py'de kalır). Colab-güvenli: her ckpt_every adımda
trainable durum + optimizer + scheduler + adım sayısı kaydedilir; resume birebir.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch


@dataclass
class TrainConfig:
    max_steps: int
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    ckpt_every: int = 500
    log_every: int = 50
    out_dir: str = "v2/checkpoints/stage1"
    extra: dict = field(default_factory=dict)


def cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


class TrainLoop:
    """
    Args:
        params: eğitilecek parametreler (LoRA + control_proj vb.).
        step_fn: batch -> skaler loss (autocast/backbone çağrısı içeride).
        state_provider/state_loader: trainable ağırlıkların ckpt sözlüğü
            (LoRA+sidecar'ı toplayıp geri yükleyen çağrılar — loop bilmez).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        step_fn: Callable[[dict], torch.Tensor],
        cfg: TrainConfig,
        *,
        state_provider: Callable[[], dict] | None = None,
        state_loader: Callable[[dict], None] | None = None,
        log=print,
    ):
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("Eğitilecek parametre yok")
        self.step_fn = step_fn
        self.cfg = cfg
        self.state_provider = state_provider
        self.state_loader = state_loader
        self.log = log
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: cosine_warmup(s, cfg.warmup_steps, cfg.max_steps)
        )
        self.step = 0

    # ------------------------------ checkpoint ------------------------------ #

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or Path(self.cfg.out_dir) / f"ckpt_{self.step:06d}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict(),
                "trainables": self.state_provider() if self.state_provider else None,
            },
            path,
        )
        (path.parent / "latest.pt").write_text(str(path))  # işaretçi dosya
        return path

    def resume(self, path: str | Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.step = ck["step"]
        self.opt.load_state_dict(ck["optimizer"])
        self.sched.load_state_dict(ck["scheduler"])
        if ck.get("trainables") is not None and self.state_loader:
            self.state_loader(ck["trainables"])
        self.log(f"resume: {path} (adım {self.step})")

    def maybe_resume(self) -> None:
        latest = Path(self.cfg.out_dir) / "latest.pt"
        if latest.exists():
            self.resume(latest.read_text().strip())

    # -------------------------------- döngü -------------------------------- #

    def run(self, loader) -> dict:
        cfg = self.cfg
        it = iter(loader)
        running, t0 = 0.0, time.time()
        while self.step < cfg.max_steps:
            self.opt.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(cfg.grad_accum):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                loss = self.step_fn(batch) / cfg.grad_accum
                loss.backward()
                accum_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
            self.opt.step()
            self.sched.step()
            self.step += 1
            running += accum_loss
            if self.step % cfg.log_every == 0:
                self.log(
                    f"adım {self.step}/{cfg.max_steps} loss={running / cfg.log_every:.4f} "
                    f"lr={self.sched.get_last_lr()[0]:.2e} {(time.time() - t0) / cfg.log_every:.2f}s/adım"
                )
                running, t0 = 0.0, time.time()
            if self.step % cfg.ckpt_every == 0 or self.step == cfg.max_steps:
                self.save()
        return {"final_step": self.step}
