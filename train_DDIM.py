import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

# Denoising diffusion implicit model DDIM

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv1d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)

        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb

class UNet1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_mlp = nn.Sequential(
            TimeEmbedding(64),
            nn.Linear(64, 128),
            nn.SiLU()
        )

        self.down1 = ConvBlock(2, 32)
        self.down2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool1d(2)

        self.mid = ConvBlock(64, 128)

        self.up1 = ConvBlock(128 + 64, 64)
        self.up2 = ConvBlock(64 + 32, 32)

        self.final = nn.Conv1d(32, 2, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t).unsqueeze(-1)

        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))

        x_mid = self.mid(self.pool(x2))

        t_emb = t_emb.expand(-1, -1, x_mid.size(2))
        x_mid = x_mid + t_emb

        x = F.interpolate(x_mid, scale_factor=2)
        x = self.up1(torch.cat([x, x2], dim=1))

        x = F.interpolate(x, scale_factor=2)
        x = self.up2(torch.cat([x, x1], dim=1))

        return self.final(x)


def cosine_beta_schedule(T):
    s = 0.008
    steps = T + 1
    x = torch.linspace(0, T, steps)

    alphas = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas = alphas / alphas[0]

    betas = 1 - (alphas[1:] / alphas[:-1])
    return torch.clip(betas, 0.0001, 0.999)

class Diffusion(nn.Module):
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model
        self.T = T

        betas = cosine_beta_schedule(T)
        alphas = 1. - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)

        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus', torch.sqrt(1 - alpha_bar))

    def q_sample(self, x0, t, noise):
        return (
            self.sqrt_alpha_bar[t][:, None, None] * x0 +
            self.sqrt_one_minus[t][:, None, None] * noise
        )

    def loss(self, x0):
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device)

        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        noise_pred = self.model(xt, t)

        return F.mse_loss(noise_pred, noise)

class DDIMSampler:
    def __init__(self, diffusion, eta=0.0):
        self.model = diffusion.model
        self.T = diffusion.T
        self.eta = eta

        self.alpha_bar = diffusion.alpha_bar

    @torch.no_grad()
    def sample(self, shape, device, steps=50):

        x = torch.randn(shape, device=device)

        time_steps = torch.linspace(self.T - 1, 0, steps, device=device).long()

        for i in range(steps - 1):
            t = time_steps[i]
            t_next = time_steps[i + 1]

            t_batch = torch.full((shape[0],), t, device=device)

            eps = self.model(x, t_batch)

            a_t = self.alpha_bar[t]
            a_next = self.alpha_bar[t_next]

            a_t = a_t.view(1, 1, 1)
            a_next = a_next.view(1, 1, 1)

            x0 = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)

            sigma = self.eta * torch.sqrt(
                (1 - a_next) / (1 - a_t + 1e-8)
                * (1 - a_t / (a_next + 1e-8))
            )

            noise = torch.randn_like(x) if self.eta > 0 else 0

            x = (
                    torch.sqrt(a_next) * x0 +
                    torch.sqrt(torch.clamp(1 - a_next - sigma ** 2, min=0)) * eps +
                    sigma * noise
            )

        return x

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints/ddim')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    data = np.load('data/datos_sinteticos/dataset_synthetic.npy')
    data = torch.from_numpy(data).float()

    loader = DataLoader(
        TensorDataset(data),
        batch_size=args.batch_size,
        shuffle=True
    )

    model = UNet1D().to(device)
    diffusion = Diffusion(model).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        loss_avg = 0

        for (x,) in loader:
            x = x.to(device)

            loss = diffusion.loss(x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_avg += loss.item()

        loss_avg /= len(loader)

        print(f"Epoch {epoch} | Loss {loss_avg:.6f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"model_{epoch}.pth"))

if __name__ == "__main__":
    train()
