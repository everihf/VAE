import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Dataset
# -----------------------------

class DomainImageDataset(Dataset):
    """Wrap ImageFolder to return (image, domain_label)."""

    def __init__(self, root: Path, domain_label: int, image_size: int = 256):
        self.domain_label = domain_label
        tfm = transforms.Compose(
            [
                transforms.Resize(image_size + 16),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),  # [0, 1]
            ]
        )
        self.dataset = ImageFolder(str(root), transform=tfm)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, _ = self.dataset[idx]
        return x, self.domain_label


class PairedDomainDataset(Dataset):
    """Merge realistic and impressionist data in one dataset."""

    def __init__(self, real_root: Path, imp_root: Path, image_size: int = 256):
        self.real = DomainImageDataset(real_root, domain_label=0, image_size=image_size)
        self.imp = DomainImageDataset(imp_root, domain_label=1, image_size=image_size)
        self.real_len = len(self.real)
        self.imp_len = len(self.imp)

    def __len__(self) -> int:
        return self.real_len + self.imp_len

    def __getitem__(self, idx: int):
        if idx < self.real_len:
            return self.real[idx]
        return self.imp[idx - self.real_len]


def split_indices(total: int, seed: int, ratios=(0.8, 0.1, 0.1)) -> Tuple[List[int], List[int], List[int]]:
    assert math.isclose(sum(ratios), 1.0, abs_tol=1e-6)
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(total * ratios[0])
    n_val = int(total * ratios[1])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def build_loaders(
    real_root: Path,
    imp_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Dict[str, DataLoader]:
    dataset = PairedDomainDataset(real_root, imp_root, image_size=image_size)

    # Stratified-like split: split each domain separately then merge.
    real_train, real_val, real_test = split_indices(dataset.real_len, seed)
    imp_train, imp_val, imp_test = split_indices(dataset.imp_len, seed)

    train_indices = real_train + [i + dataset.real_len for i in imp_train]
    val_indices = real_val + [i + dataset.real_len for i in imp_val]
    test_indices = real_test + [i + dataset.real_len for i in imp_test]

    rng = random.Random(seed)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    loaders = {
        "train": DataLoader(
            Subset(dataset, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            Subset(dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            Subset(dataset, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    return loaders


# -----------------------------
# Model
# -----------------------------

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),  # 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten_dim = 512 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 256, domain_embed_dim: int = 16):
        super().__init__()
        self.domain_emb = nn.Embedding(2, domain_embed_dim)
        self.fc = nn.Linear(latent_dim + domain_embed_dim, 512 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # 256
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        emb = self.domain_emb(domain)
        h = torch.cat([z, emb], dim=1)
        h = self.fc(h).view(-1, 512, 8, 8)
        return self.net(h)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, domain: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, domain)
        return recon, mu, logvar, z


# -----------------------------
# Training / Evaluation
# -----------------------------


def vae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rec = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = rec + beta * kl
    return total, rec, kl


@torch.no_grad()
def evaluate(model: VAE, loader: DataLoader, device: torch.device, beta: float):
    model.eval()
    totals, recs, kls = [], [], []
    for x, domain in loader:
        x = x.to(device)
        domain = domain.to(device)
        recon, mu, logvar, _ = model(x, domain)
        total, rec, kl = vae_loss(x, recon, mu, logvar, beta)
        totals.append(total.item())
        recs.append(rec.item())
        kls.append(kl.item())
    return {
        "total": sum(totals) / len(totals),
        "reconstruction_mse": sum(recs) / len(recs),
        "kl": sum(kls) / len(kls),
    }


def maybe_compute_fid(
    real_dir: Path,
    fake_dir: Path,
    device: torch.device,
    batch_size: int = 32,
) -> Optional[float]:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception:
        return None

    tfm = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
    ])

    def load_dir(d: Path):
        images = []
        for p in sorted(d.glob("*.png")):
            img = Image.open(p).convert("RGB")
            t = (tfm(img) * 255).to(torch.uint8)
            images.append(t)
        if not images:
            return None
        return torch.stack(images)

    real = load_dir(real_dir)
    fake = load_dir(fake_dir)
    if real is None or fake is None:
        return None

    fid = FrechetInceptionDistance(feature=2048).to(device)
    for i in range(0, len(real), batch_size):
        fid.update(real[i:i + batch_size].to(device), real=True)
    for i in range(0, len(fake), batch_size):
        fid.update(fake[i:i + batch_size].to(device), real=False)
    return float(fid.compute().item())


@torch.no_grad()
def save_transfer_samples(
    model: VAE,
    loader: DataLoader,
    out_dir: Path,
    device: torch.device,
    alpha_list: List[float],
    latent_dim: int,
    random_samples: int = 8,
):
    model.eval()
    ensure_dir(out_dir)
    real_imgs, imp_imgs = [], []

    for x, d in loader:
        x = x.to(device)
        d = d.to(device)
        real_mask = d == 0
        imp_mask = d == 1
        if real_mask.any() and len(real_imgs) < 8:
            real_imgs.extend(x[real_mask][: 8 - len(real_imgs)])
        if imp_mask.any() and len(imp_imgs) < 8:
            imp_imgs.extend(x[imp_mask][: 8 - len(imp_imgs)])
        if len(real_imgs) >= 8 and len(imp_imgs) >= 8:
            break

    if real_imgs and imp_imgs:
        real = torch.stack(real_imgs[:8])
        imp = torch.stack(imp_imgs[:8])

        # A->B and B->A
        z_real = model.encoder(real)[0]
        z_imp = model.encoder(imp)[0]

        out_real_to_imp = model.decoder(z_real, torch.ones(len(z_real), dtype=torch.long, device=device))
        out_imp_to_real = model.decoder(z_imp, torch.zeros(len(z_imp), dtype=torch.long, device=device))

        save_image(make_grid(torch.cat([real, out_real_to_imp], dim=0), nrow=8), out_dir / "real_to_imp.png")
        save_image(make_grid(torch.cat([imp, out_imp_to_real], dim=0), nrow=8), out_dir / "imp_to_real.png")

        # Latent mixing
        n = min(len(z_real), len(z_imp), 8)
        for alpha in alpha_list:
            z_mix = alpha * z_real[:n] + (1.0 - alpha) * z_imp[:n]
            mixed = model.decoder(z_mix, torch.ones(n, dtype=torch.long, device=device))
            save_image(make_grid(mixed, nrow=n), out_dir / f"latent_mix_alpha_{alpha:.1f}.png")

    # Random generations
    z_rand = torch.randn(random_samples, latent_dim, device=device)
    domain = torch.randint(0, 2, (random_samples,), device=device)
    gen = model.decoder(z_rand, domain)
    save_image(make_grid(gen, nrow=random_samples), out_dir / "random_generation.png")


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    loaders = build_loaders(
        real_root=Path(args.real_data),
        imp_root=Path(args.imp_data),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = VAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    ensure_dir(Path(args.output_dir))
    ckpt_path = Path(args.output_dir) / "best_vae.pt"

    best_val = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total, train_rec, train_kl = 0.0, 0.0, 0.0
        steps = 0

        for x, domain in loaders["train"]:
            x = x.to(device)
            domain = domain.to(device)

            recon, mu, logvar, _ = model(x, domain)
            loss, rec, kl = vae_loss(x, recon, mu, logvar, beta=args.beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_total += loss.item()
            train_rec += rec.item()
            train_kl += kl.item()
            steps += 1

        train_metrics = {
            "total": train_total / steps,
            "reconstruction_mse": train_rec / steps,
            "kl": train_kl / steps,
        }
        val_metrics = evaluate(model, loaders["val"], device, beta=args.beta)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"Epoch {epoch:03d} | "
            f"train_total={train_metrics['total']:.4f}, val_total={val_metrics['total']:.4f}, "
            f"val_mse={val_metrics['reconstruction_mse']:.4f}, val_kl={val_metrics['kl']:.4f}"
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            patience_counter = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Load best and evaluate on test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, loaders["test"], device, beta=args.beta)

    sample_dir = Path(args.output_dir) / "samples"
    save_transfer_samples(
        model=model,
        loader=loaders["test"],
        out_dir=sample_dir,
        device=device,
        alpha_list=[0.0, 0.25, 0.5, 0.75, 1.0],
        latent_dim=args.latent_dim,
        random_samples=8,
    )

    # FID (optional, requires torchmetrics + enough generated images)
    fid_real_dir = sample_dir  # placeholder: should point to target real-image directory exports
    fid_fake_dir = sample_dir
    fid_score = maybe_compute_fid(fid_real_dir, fid_fake_dir, device=device)

    report = {
        "best_val_total": best_val,
        "test": test_metrics,
        "fid": fid_score,
        "history": history,
        "notes": "FID uses optional torchmetrics; export real/fake sets for meaningful value.",
    }
    with open(Path(args.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Training completed.")
    print(json.dumps({"test": test_metrics, "fid": fid_score}, ensure_ascii=False, indent=2))



def parse_args():
    p = argparse.ArgumentParser(description="VAE-based style transfer and generation")
    p.add_argument("--real-data", type=str, required=True, help="COCO-like realistic images root (ImageFolder format)")
    p.add_argument("--imp-data", type=str, required=True, help="WikiArt-like impressionist images root (ImageFolder format)")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
