"""
Two-phase CRL training pipeline.

Phase 1 (--phase crl):
  Trains MultiModalCausalVAE with self-supervised objectives only.
  z_veh is trained with VICReg (invariance + variance + covariance).
  z_env is trained with KL divergence (iVAE prior) + temporal slowness.
  No class labels used. Best checkpoint saved by val loss.

Phase 2 (--phase downstream):
  Loads frozen CRL encoder; trains DetectionHead (binary) and
  ClassificationHead (4-class) simultaneously. Requires a saved CRL
  checkpoint in --save-dir.

--phase full runs both sequentially.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from causal_dataset import MultiModalCausalDataset, collate_multimodal
from causal_vae import MultiModalCausalVAE
from crl_config import (
    Z_VEH_DIM,
    Z_ENV_DIM,
    MODALITY_FEATURE_DIM,
    BATCH_SIZE,
    CRL_EPOCHS,
    DOWNSTREAM_EPOCHS,
    LEARNING_RATE,
    LAMBDA_SLOW,
    BETA_KL,
    NUM_WORKERS,
    EARLY_STOP_PATIENCE,
    LR_FACTOR,
    LR_PATIENCE,
    CLASS_MAP,
    LAMBDA_VIC_INV,
    LAMBDA_VIC_VAR,
    LAMBDA_VIC_COV,
)


# ---------------------------------------------------------------------------
# Downstream heads
# ---------------------------------------------------------------------------


class DetectionHead(nn.Module):
    """Binary: 0 = background/absent, 1 = vehicle present."""

    def __init__(self, z_veh_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_veh_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, z_veh: torch.Tensor) -> torch.Tensor:
        return self.net(z_veh)


class ClassificationHead(nn.Module):
    """4-class: pedestrian / light / sport / utility."""

    def __init__(self, z_veh_dim: int, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_veh_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, z_veh: torch.Tensor) -> torch.Tensor:
        return self.net(z_veh)


# ---------------------------------------------------------------------------
# CRL loss
# ---------------------------------------------------------------------------


def _vicreg_loss(
    z_t: torch.Tensor,
    z_next: torch.Tensor,
    lambda_inv: float,
    lambda_var: float,
    lambda_cov: float,
) -> tuple:
    """
    VICReg objective for z_veh.

    z_t, z_next: [B, D] deterministic vehicle embeddings for window t and t+1.

    Returns (vic_total, inv, var, cov).
    """
    B, D = z_t.shape

    # Invariance: same vehicle, adjacent windows should map to same z_veh
    inv = F.mse_loss(z_t, z_next)

    # Variance: per-dimension std across batch should stay above 1
    # Operates on both views independently
    def _var_loss(z):
        std = z.std(dim=0)  # [D]
        return F.relu(1.0 - std).mean()

    var = _var_loss(z_t) + _var_loss(z_next)

    # Covariance: off-diagonal entries of cov(z) should be ~0
    def _cov_loss(z):
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (B - 1)  # [D, D]
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / D

    cov = _cov_loss(z_t) + _cov_loss(z_next)

    total = lambda_inv * inv + lambda_var * var + lambda_cov * cov
    return total, inv, var, cov


def crl_loss(out: dict, beta_kl: float, lambda_slow: float) -> tuple:
    """
    Compute CRL objective.

    z_veh: VICReg (invariance + variance + covariance) — no KL, no recon.
    z_env: KL divergence against iVAE prior + temporal slowness.

    Returns (total, vic_inv, vic_var, vic_cov, kl_env, slow).
    """
    z_veh_t = out["z_veh_t"]
    z_veh_next = out["z_veh_next"]
    mu_env = out["mu_env_t"]
    logvar_env = out["logvar_env_t"]
    z_env_t = out["z_env_t"]
    z_env_next = out["z_env_next"]
    prior_mu = out["prior_mu_env"]
    prior_logvar = out["prior_logvar_env"]
    B = z_veh_t.size(0)

    # 1. VICReg on z_veh
    vic_total, vic_inv, vic_var, vic_cov = _vicreg_loss(
        z_veh_t,
        z_veh_next,
        LAMBDA_VIC_INV,
        LAMBDA_VIC_VAR,
        LAMBDA_VIC_COV,
    )

    # 2. KL divergence — environment (sensor-domain conditional prior)
    logvar_diff = prior_logvar - logvar_env
    kl_env = (
        0.5
        * torch.sum(
            logvar_diff
            - 1
            + (logvar_env.exp() + (mu_env - prior_mu).pow(2)) / prior_logvar.exp()
        )
        / B
    )

    # 3. Temporal slow loss — z_env only
    slow = F.mse_loss(z_env_t, z_env_next) * lambda_slow

    total = vic_total + beta_kl * kl_env + slow
    return total, vic_inv, vic_var, vic_cov, kl_env, slow


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"Using NVIDIA GPU (CUDA): {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        d = torch.device("cpu")
        print("WARNING: No GPU detected, using CPU.")
    return d


# ---------------------------------------------------------------------------
# Phase 1: CRL pre-training
# ---------------------------------------------------------------------------


def train_crl_phase(
    model: MultiModalCausalVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    beta_kl: float,
    lambda_slow: float,
    save_dir: Path,
    device: torch.device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_FACTOR, patience=LR_PATIENCE
    )
    best_val_loss = float("inf")
    patience_ctr = 0

    metrics_path = save_dir / "crl_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "vic_inv",
                "vic_var",
                "vic_cov",
                "kl_env",
                "slow",
            ]
        )

    print("\n=== Phase 1: CRL Pre-training ===")
    for epoch in range(epochs):
        model.train()
        epoch_loss = epoch_inv = epoch_var = epoch_cov = 0.0
        epoch_kl_env = epoch_slow = 0.0

        # KL annealing on z_env: ramp from 0.1 to 1.0 over first 10 epochs
        beta_kl_annealed = min(1.0, 0.1 + epoch / 10.0) * beta_kl

        for batch_idx, batch in enumerate(train_loader):
            batch_t, batch_next, avail, domain_ids, _, _ = batch

            batch_t = {
                m: v.to(device) if v is not None else None for m, v in batch_t.items()
            }
            batch_next = {
                m: v.to(device) if v is not None else None
                for m, v in batch_next.items()
            }
            avail = avail.to(device)
            domain_ids = domain_ids.to(device)

            optimizer.zero_grad()
            out = model(batch_t, batch_next, avail, domain_ids)
            loss, vic_inv, vic_var, vic_cov, kl_e, slow = crl_loss(
                out, beta_kl_annealed, lambda_slow
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_inv += vic_inv.item()
            epoch_var += vic_var.item()
            epoch_cov += vic_cov.item()
            epoch_kl_env += kl_e.item()
            epoch_slow += slow.item()

            if batch_idx % 50 == 0:
                print(
                    f"  Epoch {epoch} | Batch {batch_idx} | "
                    f"Inv: {vic_inv:.3f} | Var: {vic_var:.3f} | "
                    f"Cov: {vic_cov:.3f} | KL_env: {kl_e:.3f} | "
                    f"Slow: {slow:.3f}"
                )

        n = len(train_loader)
        train_avg = epoch_loss / n

        # Validation
        val_loss = _eval_crl(model, val_loader, device, beta_kl, lambda_slow)
        scheduler.step(val_loss)

        print(f"==== Epoch {epoch} | Train: {train_avg:.4f} | Val: {val_loss:.4f} ====")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    train_avg,
                    val_loss,
                    epoch_inv / n,
                    epoch_var / n,
                    epoch_cov / n,
                    epoch_kl_env / n,
                    epoch_slow / n,
                ]
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), save_dir / "crl_best.pth")
            print(f"  ✓ New best CRL model (val_loss={val_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    print(f"CRL pre-training complete. Best val loss: {best_val_loss:.4f}")


@torch.no_grad()
def _eval_crl(
    model: MultiModalCausalVAE,
    loader: DataLoader,
    device: torch.device,
    beta_kl: float,
    lambda_slow: float,
) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        batch_t, batch_next, avail, domain_ids, _, _ = batch
        batch_t = {
            m: v.to(device) if v is not None else None for m, v in batch_t.items()
        }
        batch_next = {
            m: v.to(device) if v is not None else None for m, v in batch_next.items()
        }
        avail = avail.to(device)
        domain_ids = domain_ids.to(device)
        out = model(batch_t, batch_next, avail, domain_ids)
        loss, *_ = crl_loss(out, beta_kl, lambda_slow)
        total += loss.item()
    model.train()
    return total / len(loader)


# ---------------------------------------------------------------------------
# Phase 2: Downstream head training
# ---------------------------------------------------------------------------


def train_downstream_phase(
    model: MultiModalCausalVAE,
    det_head: DetectionHead,
    cls_head: ClassificationHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    save_dir: Path,
    device: torch.device,
):
    # Freeze CRL encoder
    for param in model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        list(det_head.parameters()) + list(cls_head.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_FACTOR, patience=LR_PATIENCE
    )

    # Class weights for imbalanced categories (detection: background heavy)
    det_weight = torch.tensor([1.0, 5.0], device=device)  # upweight vehicle
    cls_weight = torch.tensor(
        [1.0, 1.0, 1.0, 1.0], device=device
    )  # update after profiling

    det_criterion = nn.CrossEntropyLoss(weight=det_weight)
    cls_criterion = nn.CrossEntropyLoss(weight=cls_weight)

    best_val_f1 = 0.0
    patience_ctr = 0

    metrics_path = save_dir / "downstream_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "train_det_loss",
                "train_cls_loss",
                "val_det_acc",
                "val_det_f1",
                "val_cls_acc",
                "val_cls_f1",
            ]
        )

    print("\n=== Phase 2: Downstream Head Training ===")
    for epoch in range(epochs):
        model.eval()
        det_head.train()
        cls_head.train()

        epoch_det_loss = epoch_cls_loss = 0.0
        n_det = n_cls = 0

        for batch in train_loader:
            (batch_t, _, avail, domain_ids, cat_labels, det_labels) = batch

            batch_t = {
                m: v.to(device) if v is not None else None for m, v in batch_t.items()
            }
            avail = avail.to(device)
            domain_ids = domain_ids.to(device)
            cat_labels = cat_labels.to(device)
            det_labels = det_labels.to(device)

            with torch.no_grad():
                z_veh = model.encode_veh(batch_t, avail)

            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)

            # Detection loss — all samples
            det_logits = det_head(z_veh)
            det_loss = det_criterion(det_logits, det_labels)
            loss = loss + det_loss
            epoch_det_loss += det_loss.item()
            n_det += 1

            # Classification loss — exclude background and multi-vehicle
            cls_mask = cat_labels >= 0
            if cls_mask.any():
                cls_logits = cls_head(z_veh[cls_mask])
                cls_labels = cat_labels[cls_mask]
                cls_loss = cls_criterion(cls_logits, cls_labels)
                loss = loss + cls_loss
                epoch_cls_loss += cls_loss.item()
                n_cls += 1

            loss.backward()
            optimizer.step()

        val_metrics = _eval_downstream(model, det_head, cls_head, val_loader, device)
        val_cls_f1 = val_metrics["cls_f1"]
        scheduler.step(-val_cls_f1)

        print(
            f"Epoch {epoch} | "
            f"Det loss: {epoch_det_loss / max(n_det,1):.4f} | "
            f"Cls loss: {epoch_cls_loss / max(n_cls,1):.4f} | "
            f"Val det F1: {val_metrics['det_f1']:.4f} | "
            f"Val cls F1: {val_cls_f1:.4f}"
        )

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    epoch_det_loss / max(n_det, 1),
                    epoch_cls_loss / max(n_cls, 1),
                    val_metrics["det_acc"],
                    val_metrics["det_f1"],
                    val_metrics["cls_acc"],
                    val_cls_f1,
                ]
            )

        if val_cls_f1 > best_val_f1:
            best_val_f1 = val_cls_f1
            patience_ctr = 0
            torch.save(det_head.state_dict(), save_dir / "det_head_best.pth")
            torch.save(cls_head.state_dict(), save_dir / "cls_head_best.pth")
            print(f"  ✓ New best (val cls F1={val_cls_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    print(f"Downstream training complete. Best val cls F1: {best_val_f1:.4f}")


@torch.no_grad()
def _eval_downstream(
    model: MultiModalCausalVAE,
    det_head: DetectionHead,
    cls_head: ClassificationHead,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    det_head.eval()
    cls_head.eval()

    det_preds, det_true = [], []
    cls_preds, cls_true = [], []

    for batch in loader:
        batch_t, _, avail, _, cat_labels, det_labels = batch
        batch_t = {
            m: v.to(device) if v is not None else None for m, v in batch_t.items()
        }
        avail = avail.to(device)
        z_veh = model.encode_veh(batch_t, avail)

        det_preds.extend(det_head(z_veh).argmax(dim=1).cpu().tolist())
        det_true.extend(det_labels.tolist())

        cls_mask = cat_labels >= 0
        if cls_mask.any():
            cls_preds.extend(
                cls_head(z_veh[cls_mask.to(device)]).argmax(dim=1).cpu().tolist()
            )
            cls_true.extend(cat_labels[cls_mask].tolist())

    det_acc, det_f1 = _acc_f1(det_true, det_preds)
    cls_acc, cls_f1 = _acc_f1(cls_true, cls_preds) if cls_true else (0.0, 0.0)

    model.train()
    det_head.train()
    cls_head.train()
    return {"det_acc": det_acc, "det_f1": det_f1, "cls_acc": cls_acc, "cls_f1": cls_f1}


def _acc_f1(true, pred) -> tuple:
    from sklearn.metrics import accuracy_score, f1_score

    if not true:
        return 0.0, 0.0
    return (
        accuracy_score(true, pred),
        f1_score(true, pred, average="weighted", zero_division=0),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="CRL Training Pipeline")
    p.add_argument("--phase", choices=["crl", "downstream", "full"], default="full")
    p.add_argument("--data-dir", default="./parsed/train")
    p.add_argument("--val-dir", default="./parsed/val")
    p.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Modalities to include (default: all available)",
    )
    p.add_argument("--crl-epochs", type=int, default=CRL_EPOCHS)
    p.add_argument("--downstream-epochs", type=int, default=DOWNSTREAM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--lambda-slow", type=float, default=LAMBDA_SLOW)
    p.add_argument("--beta-kl", type=float, default=BETA_KL)
    p.add_argument("--z-veh-dim", type=int, default=Z_VEH_DIM)
    p.add_argument("--z-env-dim", type=int, default=Z_ENV_DIM)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--save-dir", default="./saved_crl")
    return p.parse_args()


def build_loaders(
    data_dir,
    val_dir,
    batch_size,
    num_workers,
    modalities,
    filter_present=False,
    domain_map=None,
):
    train_ds = MultiModalCausalDataset(
        parquet_dir=data_dir,
        filter_present=filter_present,
        include_modalities=modalities,
        domain_map=domain_map,
    )
    val_ds = MultiModalCausalDataset(
        parquet_dir=val_dir,
        filter_present=filter_present,
        include_modalities=modalities,
        domain_map=train_ds._domain_to_id,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.num_sensor_domains, train_ds._domain_to_id


def main():
    args = parse_args()
    device = get_device()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    meta_path = save_dir / "crl_meta.json"

    if args.phase in ("crl", "full"):
        train_loader, val_loader, num_domains, _ = build_loaders(
            args.data_dir,
            args.val_dir,
            args.batch_size,
            args.num_workers,
            args.modalities,
            filter_present=True,
            domain_map=None,
        )
        with open(meta_path, "w") as f:
            json.dump({"num_domains": num_domains}, f)
    else:
        if meta_path.exists():
            with open(meta_path, "r") as f:
                num_domains = json.load(f)["num_domains"]
        else:
            raise FileNotFoundError(f"Missing {meta_path}. Run --phase crl first.")

        train_loader, val_loader, _, _ = build_loaders(
            args.data_dir,
            args.val_dir,
            args.batch_size,
            args.num_workers,
            args.modalities,
            filter_present=False,
            domain_map={"__UNKNOWN__": 0},
        )

    model = MultiModalCausalVAE(
        num_sensor_domains=num_domains,
        modality_feat_dim=MODALITY_FEATURE_DIM,
        z_veh_dim=args.z_veh_dim,
        z_env_dim=args.z_env_dim,
    ).to(device)

    det_head = DetectionHead(z_veh_dim=args.z_veh_dim).to(device)
    cls_head = ClassificationHead(
        z_veh_dim=args.z_veh_dim, num_classes=len(CLASS_MAP)
    ).to(device)

    if args.phase in ("crl", "full"):
        train_crl_phase(
            model,
            train_loader,
            val_loader,
            epochs=args.crl_epochs,
            lr=args.lr,
            beta_kl=args.beta_kl,
            lambda_slow=args.lambda_slow,
            save_dir=save_dir,
            device=device,
        )

    if args.phase in ("downstream", "full"):
        if args.phase == "full":
            print(
                "\nRebuilding data loaders with filter_present=False for downstream phase..."
            )
            train_loader, val_loader, _, _ = build_loaders(
                args.data_dir,
                args.val_dir,
                args.batch_size,
                args.num_workers,
                args.modalities,
                filter_present=False,
                domain_map={"__UNKNOWN__": 0},
            )

        crl_ckpt = save_dir / "crl_best.pth"
        if crl_ckpt.exists():
            model.load_state_dict(torch.load(crl_ckpt, map_location=device))
            print(f"Loaded CRL checkpoint from {crl_ckpt}")
        elif args.phase == "downstream":
            raise FileNotFoundError(
                f"No CRL checkpoint at {crl_ckpt}. Run --phase crl first."
            )

        train_downstream_phase(
            model,
            det_head,
            cls_head,
            train_loader,
            val_loader,
            epochs=args.downstream_epochs,
            lr=args.lr,
            save_dir=save_dir,
            device=device,
        )

    # Save final model + head weights alongside best checkpoints
    torch.save(model.state_dict(), save_dir / "crl_final.pth")
    torch.save(det_head.state_dict(), save_dir / "det_head_final.pth")
    torch.save(cls_head.state_dict(), save_dir / "cls_head_final.pth")
    print(f"\nAll weights saved to {save_dir}/")


if __name__ == "__main__":
    main()
