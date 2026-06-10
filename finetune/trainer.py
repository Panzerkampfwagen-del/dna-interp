"""Fine-tuning loop for the enhancer and TF-binding tasks.

Backbone and classification head get separate learning rates via parameter
groups, a linear-warmup then cosine-decay schedule, BF16 autocast, optional
gradient accumulation, validation-metric early stopping, and best-checkpoint
saving. BF16 needs no GradScaler, so the loop stays simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from config import CHECKPOINTS, autocast_dtype, set_seed
from finetune.evaluate import evaluate


@dataclass
class TrainConfig:
    task: str = "enhancer"  # "enhancer" or "tf_binding"
    epochs: int = 5
    lr_backbone: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    accum_steps: int = 1
    log_every: int = 50
    patience: int = 2
    use_amp: bool = True
    seed: int = 0
    out_name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.out_name:
            self.out_name = f"{self.task}_best.pt"


def _param_groups(model: nn.Module, cfg: TrainConfig) -> list[dict]:
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith("classifier") else backbone).append(p)
    return [
        {"params": backbone, "lr": cfg.lr_backbone},
        {"params": head, "lr": cfg.lr_head},
    ]


def _loss_fn(task: str):
    return nn.BCEWithLogitsLoss() if task == "tf_binding" else nn.CrossEntropyLoss()


def _val_metric(metrics: dict, task: str) -> float:
    return metrics["mean_auroc"] if task == "tf_binding" else metrics["mcc"]


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainConfig,
) -> dict:
    """Fine-tune `model`, early-stopping on the task's validation metric.

    Returns a history dict and writes the best checkpoint to results/checkpoints.
    """
    set_seed(cfg.seed)
    model.to(device)
    loss_fn = _loss_fn(cfg.task)
    optim = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // cfg.accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_frac))
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)

    amp_dtype = autocast_dtype()
    use_amp = cfg.use_amp and device.type == "cuda"

    history: dict = {"loss": [], "val_metric": [], "best": None}
    best_metric = -float("inf")
    epochs_no_improve = 0
    ckpt_path = CHECKPOINTS / cfg.out_name
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        optim.zero_grad()
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(input_ids, attention_mask, cache_activations=False)
                loss = loss_fn(logits.float(), labels) / cfg.accum_steps
            loss.backward()

            if (i + 1) % cfg.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1
                if global_step % cfg.log_every == 0:
                    vm = _val_metric(evaluate(model, val_loader, device, cfg.task), cfg.task)
                    model.train()
                    history["loss"].append(float(loss.item() * cfg.accum_steps))
                    history["val_metric"].append(vm)
                    print(f"epoch {epoch} step {global_step} loss {loss.item()*cfg.accum_steps:.4f} val {vm:.4f}")

        metrics = evaluate(model, val_loader, device, cfg.task)
        vm = _val_metric(metrics, cfg.task)
        print(f"[epoch {epoch}] val {metrics}")
        if vm > best_metric:
            best_metric = vm
            epochs_no_improve = 0
            save_checkpoint(model, ckpt_path, {"epoch": epoch, "val_metric": vm, "task": cfg.task})
            history["best"] = {"epoch": epoch, "val_metric": vm}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"early stopping at epoch {epoch}")
                break

    return history


def save_checkpoint(model: nn.Module, path: Path, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device | None = None) -> dict:
    blob = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(blob["state_dict"])
    return blob["meta"]
