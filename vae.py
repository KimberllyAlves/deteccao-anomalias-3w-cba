"""Variational Autoencoder usado para modelar o estado normal dos poços."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import DEVICE, LOGVAR_CLAMP, EPOCHS, LR, BATCH_SIZE, PATIENCE, GRAD_CLIP


def sanitizar(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def para_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(sanitizar(arr)).to(DEVICE)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -LOGVAR_CLAMP, LOGVAR_CLAMP)
        return self.dec(self.reparameterize(mu, logvar)), mu, logvar


def perda_vae(recon, x, mu, logvar):
    mse = nn.functional.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld


def treinar_vae(X_np, input_dim, hidden_dim, latent_dim):
    """Treina um VAE do zero em X_np, com early stopping em cima da própria loss."""
    modelo = VAE(input_dim, hidden_dim, latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=LR)
    t = para_tensor(X_np)
    loader = DataLoader(TensorDataset(t), batch_size=BATCH_SIZE, shuffle=True)

    melhor_loss, sem_melhora, melhor_estado = float('inf'), 0, None

    modelo.train()
    for _ in range(EPOCHS):
        loss_ep = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, mu, logvar = modelo(batch)
            loss = perda_vae(recon, batch, mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_ep += loss.item()

        if not np.isfinite(loss_ep):
            break
        if loss_ep < melhor_loss - 1e-4:
            melhor_loss, sem_melhora = loss_ep, 0
            melhor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= PATIENCE:
                break

    if melhor_estado:
        modelo.load_state_dict(melhor_estado)

    modelo.eval()
    with torch.no_grad():
        recon_tr, _, _ = modelo(t)
        erros_treino = torch.mean((t - recon_tr) ** 2, dim=1).cpu().numpy()

    mus_treino = extrair_mus(modelo, X_np)
    return modelo, erros_treino, mus_treino


def extrair_mus(modelo, X_np):
    t = para_tensor(X_np)
    modelo.eval()
    with torch.no_grad():
        h = modelo.enc(t)
        mu = modelo.fc_mu(h)
    return mu.cpu().numpy()


def calcular_erros_mse(modelo, X_np):
    t = para_tensor(X_np)
    modelo.eval()
    with torch.no_grad():
        recon, _, _ = modelo(t)
        return torch.mean((t - recon) ** 2, dim=1).cpu().numpy()


def calcular_mahalanobis(mus: np.ndarray, mu_ref: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = mus - mu_ref
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, cov_inv, diff))
