import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

# Denoising diffusion probabilistic model DDIM

class ConvBlock(nn.Module):
    """
    Bloque convolucional 1D básico usado en la U-Net.
    Arquitectura:
    Conv1D -> BatchNorm -> SiLU
    Conv1D -> BatchNorm -> SiLU
    Este bloque se encarga de extraer características locales
    de la señal (patrones temporales del canal).
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # dos convoluciones 1D consecutivas
        # kernel_size=3 y padding=1 preservan la longitud de la señal
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_c),
            nn.SiLU(),
            nn.Conv1d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_c),
            nn.SiLU()
        )

    def forward(self, x):
        # x: (batch, channels, length)
        return self.net(x)

class TimeEmbedding(nn.Module):
    """
    Embedding sinusoidal del timestep t.
    Implementa la codificación temporal introducida en DDPM,
    similar a la usada en Transformers.
    El objetivo es convertir el timestep discreto t en un vector
    continuo que indique al modelo "cuánto ruido" contiene la señal.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
         t: tensor de forma (batch,)
            representa el paso de difusión (nivel de ruido)
        """
        half = self.dim // 2
        # Frecuencias logarítmicamente espaciadas
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        # Proyección sinusoidal
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        # salida: (batch, dim)
        return emb

class UNet1D(nn.Module):
    """
    U-Net 1D simplificada para modelar señales de canal.

    El modelo aprende a predecir el ruido ε añadido a la señal
    en un determinado timestep t.
    """
    def __init__(self, L=128):
        super().__init__()
        # MLP que transforma el timestep t en un embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(64),
            nn.Linear(64, 128),
            nn.SiLU()
        )
        # Encoder (downsampling)
        # extrae características a distintas escalas temporales
        self.down1 = ConvBlock(2, 32)
        self.down2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool1d(2)
        # Bottleneck
        # parte central donde se inyecta la información temporal
        self.mid = ConvBlock(64, 128)
        # Decoder (unsampling)
        # reconstruye la señal combinando información fina y gruesa
        self.up1 = ConvBlock(128 + 64, 64)
        self.up2 = ConvBlock(64 + 32, 32)
        # Capa final: predicción del ruido (misma dimensión que la señal)
        self.final = nn.Conv1d(32, 2, 1)

    def forward(self, x, t):
        """
        x: señal ruidosa x_t -> (batch, 1, L)
        t: timestep -> (batch,)
        """
        # Obtener embedding temporal y adaptarlo a 1D
        t_emb = self.time_mlp(t).unsqueeze(-1)
        # Encoder
        x1 = self.down1(x) # nivel 1
        x2 = self.down2(self.pool(x1)) # nivel 2
        # Bottleneck
        x_mid = self.mid(self.pool(x2))
        # expandir embeddings temporal a la dimensión temporal de x_mid
        t_emb = self.time_mlp(t).unsqueeze(-1)
        t_emb = t_emb.expand(-1, -1, x_mid.size(2))
        # Inyección del tiempo (condicionamiento temporal)
        x_mid = x_mid + t_emb
        # Decoder
        x = F.interpolate(x_mid, scale_factor=2)
        x = self.up1(torch.cat([x, x2], dim=1))

        x = F.interpolate(x, scale_factor=2)
        x = self.up2(torch.cat([x, x1], dim=1))
        # predicción final del ruido ε
        return self.final(x)

def cosine_beta_schedule(T):
    """
    Cosine noise schedule propuesto por Nichol & Dhariwal (2021).
    Define cuánto ruido se añade en cada paso de difusión.
    Produce un proceso más estable que el schedule lineal.
    """
    s = 0.008
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas = alphas / alphas[0]
    # beta_t = 1 - alpha_t / alpha_{t-1}
    betas = 1 - (alphas[1:] / alphas[:-1])
    return torch.clip(betas, 0.0001, 0.999)

class DDPM(nn.Module):
    """
    Implementación básica de DDPM para señales 1D.

    Encapsula:
    - proceso forward (añadir ruido)
    - función de pérdida
    """
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model
        self.T = T
        # Definición del proceso de difusión
        betas = cosine_beta_schedule(T)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Guardar constantes como buffers (no entrenables)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alpha', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus', torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, x0, t, noise):
        """
        Aplicar el proceso forward:
        x_t = sqrt(alpha_bar_t) * x_0 +
            sqrt(1 - alpha_bar_t) * ε
        """
        return (
            self.sqrt_alpha[t][:, None, None] * x0 +
            self.sqrt_one_minus[t][:, None, None] * noise
        )

    def loss(self, x0):
        """
        Función de pérdida DDPM:
        1. Elegir un timestep t aleatorio
        2. Añadir ruido gaussiano
        3. Predecir el ruido con al U-Net
        4. Minimizar MSE entre ruido y real
        """
        b = x0.size(0)
        # timestep aleatorio por muestra
        t = torch.randint(0, self.T, (b,), device=x0.device)
        # ruido gaussiano ε ~ N(0, I)
        noise = torch.randn_like(x0)
        # señal ruidosa x_t
        xt = self.q_sample(x0, t, noise)
        # predicción del ruido
        noise_pred = self.model(xt, t)
        # MSE loss
        return F.mse_loss(noise_pred, noise)

def train():
    """
    Bucle principal de entrenamiento del DDPM.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints/ddpm')
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    # cargar dataset
    data = np.load('data/datos_sinteticos/dataset_synthetic.npy')
    data = torch.from_numpy(data).float()

    loader = DataLoader(
        TensorDataset(data),
        batch_size=args.batch_size,
        shuffle=True
    )

    model = UNet1D().to(device)
    ddpm = DDPM(model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    epochs_no_improve = 0
    min_delta = 1e-4

    for epoch in range(1, args.epochs + 1):
        loss_avg = 0.0

        for (x,) in loader:
            x = x.to(device)
            loss = ddpm.loss(x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_avg += loss.item()

        loss_avg /= len(loader)
        print(f"Epoch {epoch} | Loss {loss_avg:.6f}")

        if loss_avg < best_loss - min_delta:
            best_loss = loss_avg
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, "best_model.pth"))
            print(f"Mejor modelo guardado (loss={best_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"Sin mejora durante {epochs_no_improve} epochs")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping: no ha habido mejora en {args.patience} epochs")
            break

        if epoch % 20 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'model_e{epoch}.pth'))

if __name__ == "__main__":
    train()
