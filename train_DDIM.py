import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

# Denoising diffusion implicit model DDIM

# Extrae características locales de la señal 1D
# GroupNorm estabiliza el entrenamiento
# SiLU introduce no linealidad suave
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

# Convierte el timestep t en un vector numérico continuo
# el modelo necesita saber cuánto ruido hay en la señal
# se usa codificación sinusoidal: sin(wt), cos(wt)
# que permite representar cualquier paso de difusión de forma suave
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2

        # frecuencias exponenciales
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)

        # codificación sinusoidal
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return emb

# encoder-decoder con skip connections:
# reduce dimensión -> aprende estrutura global
# recupera dimensión -> reconstruye señal
# entrada: señal ruidosa xt
# salida: ruido estimado epsilon
class UNet1D(nn.Module):
    def __init__(self):
        super().__init__()

        # embedding del tiempo
        self.time_mlp = nn.Sequential(
            TimeEmbedding(64),
            nn.Linear(64, 128),
            nn.SiLU()
        )

        # encoder
        self.down1 = ConvBlock(2, 32)
        self.down2 = ConvBlock(32, 64)
        # reduce resolución
        self.pool = nn.MaxPool1d(2)

        # capa intermedia (representación comprimida)
        self.mid = ConvBlock(64, 128)

        # decoder
        self.up1 = ConvBlock(128 + 64, 64)
        self.up2 = ConvBlock(64 + 32, 32)

        # salida: predicción del ruido
        self.final = nn.Conv1d(32, 2, 1)

    def forward(self, x, t):
        # embedding del timestep
        t_emb = self.time_mlp(t).unsqueeze(-1)

        # encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))

        # representación comprimida
        x_mid = self.mid(self.pool(x2))

        # añadir información del tiempo
        t_emb = t_emb.expand(-1, -1, x_mid.size(2))

        x_mid = x_mid + t_emb

        # decoder
        x = F.interpolate(x_mid, scale_factor=2)

        x = self.up1(torch.cat([x, x2], dim=1))

        x = F.interpolate(x, scale_factor=2)

        x = self.up2(torch.cat([x, x1], dim=1))

        return self.final(x)

# SCHEDULE DE RUIDO
# define cuánto ruido se añade en cada paso t
# beta_t = varianza del ruido añadido en el paso t
# cosine schedule: añade poco ruido al principio, añade más ruido al final.
# Mejora estabilidad frente a schedule lineal
def cosine_beta_schedule(T):
    s = 0.008
    steps = T + 1
    x = torch.linspace(0, T, steps)

    alphas = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas = alphas / alphas[0]

    betas = 1 - (alphas[1:] / alphas[:-1])
    return torch.clip(betas, 0.0001, 0.999)

# implementa el proceso forward: q(xt | x0)
# añade ruido progresivamente a la señal
class Diffusion(nn.Module):
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model

        # número total de pasos de difusión
        self.T = T

        betas = cosine_beta_schedule(T)
        alphas = 1. - betas

        # producto acumulado
        alpha_bar = torch.cumprod(alphas, dim=0)

        # guardar constantes
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)

        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus', torch.sqrt(1 - alpha_bar))

    # FORWARD DIFFUSION
    # genera xt a partir de x0
    # mezcla señal original + ruido gaussiano
    def q_sample(self, x0, t, noise):
        return (
            self.sqrt_alpha_bar[t][:, None, None] * x0 +
            self.sqrt_one_minus[t][:, None, None] * noise
        )

    # FUNCIÓN DE PÉRDIDA
    # la red aprende a predecir el ruido añadido
    # objetivo: epsilon_theta(xt, t) = epsilon
    def loss(self, x0):
        b = x0.size(0)

        # elegir paso aleatorio
        t = torch.randint(0, self.T, (b,), device=x0.device)

        # ruido gaussiano
        noise = torch.randn_like(x0)

        # generar señal ruidosa
        xt = self.q_sample(x0, t, noise)

        # predicción del ruido
        noise_pred = self.model(xt, t)

        # minimizar error entre ruido real y estimado
        return F.mse_loss(noise_pred, noise)

# genera nuevas señales a partir de ruido puro
# DDIm permite usar menos pasos
# usa una ecuación determinista
class DDIMSampler:
    def __init__(self, diffusion, eta=0.0):
        self.model = diffusion.model
        self.T = diffusion.T

        # controla aleatoriedad, eta=0 proceso determinista
        self.eta = eta
        self.alpha_bar = diffusion.alpha_bar

    @torch.no_grad()
    def sample(self, shape, device, steps=50):
        # comenzar desde ruido gaussiano puro
        x = torch.randn(shape, device=device)

        # seleccionar subconjunto de timesteps
        time_steps = torch.linspace(self.T - 1, 0, steps, device=device).long()

        for i in range(len(time_steps) - 1):
            t = time_steps[i]
            t_next = time_steps[i + 1]

            t_batch = torch.full((shape[0],), t, device=device)

            # estimar ruido presente en la señal
            eps = self.model(x, t_batch)

            a_t = self.alpha_bar[t].view(1, 1, 1)
            a_next = self.alpha_bar[t_next].view(1, 1, 1)

            # estimación de la señal limpia x0 = (xt - ruido) / escala
            x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)
            x0_pred = torch.clamp(x0_pred, -1, 1)

            # controla cantidad de ruido extra
            sigma = self.eta * torch.sqrt(
                (1 - a_next) / (1 - a_t + 1e-8)
                * (1 - a_t / (a_next + 1e-8))
            )

            noise = torch.randn_like(x) if self.eta > 0 else 0

            # ecuación DDIM: combina señal estimada limpia, dirección del ruido y ruido opcional
            x = (
                    torch.sqrt(a_next) * x0_pred +
                    torch.sqrt(torch.clamp(1 - a_next - sigma ** 2, min=0)) * eps +
                    sigma * noise
            )

        # último paso: señal limpia
        # en t=0: extrae x0 limpio
        t_last = time_steps[-1]
        t_batch = torch.full((shape[0],), t_last, device=device)
        eps = self.model(x, t_batch)
        a_t = self.alpha_bar[t_last].view(1, 1, 1)
        x0_final = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t + 1e-8)

        return torch.clamp(x0_final, -1, 1)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        loss_avg = 0

        for (x,) in loader:
            x = x.to(device)

            loss = diffusion.loss(x)

            opt.zero_grad()
            loss.backward()

            # evita el nan por exploding gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()
            loss_avg += loss.item()

        loss_avg /= len(loader)
        scheduler.step()

        print(f"Epoch {epoch} | Loss {loss_avg:.6f}")

        if loss_avg < best_loss:
            best_loss = loss_avg
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, "best_model.pth"))

        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"model_{epoch}.pth"))

if __name__ == "__main__":
    train()