import argparse
import os
import numpy as np
import torch
from sympy.physics.pring import energy
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# ----- GENERATOR Conv1D ----- #
class Generator(nn.Module):
    def __init__(self, z_dim, L):
        super().__init__()
        self.init_len = L // 8
        self.fc = nn.Linear(z_dim, 128 * self.init_len)
        self.bn0 = nn.BatchNorm1d(128)

        self.net = nn.Sequential(
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, self.init_len)
        x = self.bn0(x)
        x = self.net(x)
        # energy = torch.sqrt(torch.sum(x ** 2, dim=2, keepdim=True))
        # x = x / (energy + 1e-12)
        return x

# ----- DISCRIMINATOR Conv1D ----- #
class Discriminator(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(128 * (L // 8), 1)

    def forward(self, x):
        f = self.net(x)
        f = f.view(x.size(0), -1)
        return self.fc(f)

### Inicialización WGAN ###
def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def gradient_penalty(critic, real, fake, device):
    bsize = real.size(0)
    epsilon = torch.rand(bsize, 1, 1, device=device)
    epsilon = epsilon.expand_as(real)

    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    d_interpolated = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(bsize, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def compute_pdp(x):
    return torch.mean(x ** 2, dim=0)

def train_gan(G, C, loader, device, epochs=100, lr=1e-4,
              λ_gp=10, n_critic=5,z_dim=128,
              patience=15, min_delta=1e-4,
              save_dir='checkpoints/WGAN_Conv1D'):

    os.makedirs(save_dir, exist_ok=True)
    # optimizadores Adam para G y D
    optC = torch.optim.Adam(C.parameters(), lr=lr, betas=(0.0, 0.9))
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))

    best_w_dist = -float("inf")
    epochs_no_improve = 0

    ### Bucle del ENTRENAMIENTO principal ###
    for epoch in range(1, epochs + 1):
        c_loss_epoch = 0.0
        g_loss_epoch = 0.0
        w_dist_epoch = 0.0

        for (real_batch,) in loader:
            real = real_batch.to(device)
            bsize = real.size(0)

            ### Entrenar Critic ###
            for _ in range(n_critic):
                z = torch.randn(bsize, z_dim, device=device)
                fake = G(z).detach()

                real_score = C(real).mean()
                fake_score = C(fake).mean()
                loss_C = -(real_score - fake_score)
                gp = gradient_penalty(C, real, fake, device)

                loss_total = loss_C + λ_gp * gp

                optC.zero_grad()
                loss_total.backward()
                optC.step()

            # después de entrenar el critic, usamos los mismo batch de real/fake para W_dist
            with torch.no_grad():
                z = torch.randn(bsize, z_dim, device=device)
                fake_for_wdist = G(z)
                w_dist_epoch += (C(real).mean() - C(fake_for_wdist).mean()).item()

            ### Entrenar Generador ###
            z = torch.randn(bsize, z_dim, device=device)
            fake = G(z)
            loss_G = -C(fake).mean()

            optG.zero_grad()
            loss_G.backward()
            optG.step()

            c_loss_epoch += loss_C.item()
            g_loss_epoch += loss_G.item()

        # Promedios
        c_loss_epoch /= len(loader)
        g_loss_epoch /= len(loader)
        w_dist_epoch /= len(loader)

        print(f"Epoch {epoch}/{epochs} | "
              f"W_dist={w_dist_epoch:.4f} | "
              f"C_loss={c_loss_epoch:.4f} | "
              f"G_loss={g_loss_epoch:.4f}")

        if w_dist_epoch > best_w_dist + min_delta:
            best_w_dist = w_dist_epoch
            epochs_no_improve = 0

            torch.save({
                'G': G.state_dict(),
                'C': C.state_dict(),
                'epoch': epoch,
                'W_dist': best_w_dist
            }, os.path.join(save_dir, 'best_model.pth'))

            print("Nuevo mejor modelo guardado")

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping en epoch {epoch}")
            break

    print("Entrenamiento completado. Modelos guardados en", save_dir)

if __name__ == "__main__":
    ### Parámetros de ejecución ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/datos_sinteticos/dataset_synthetic.npy', help='Ruta del dataset .npy')
    parser.add_argument('--epochs', type=int, default=150, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamaño del batch')
    parser.add_argument('--z_dim', type=int, default=64, help='Dimensión del vector de ruido del generador')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (tasa de aprendizaje)')
    parser.add_argument('--n_critic', type=int, default=7,help='Número de updates del critic por update del generator')
    parser.add_argument('--lambda_gp', type=float,default=5.0, help='Peso del gradient penalty')
    parser.add_argument('--patience', type=int, default=40,help='Paciencia para early stopping')
    parser.add_argument('--min_delta', type=float,default=1e-3,help='Mejora mínima en Wasserstein distance')
    parser.add_argument('--save_dir', type=str, default='checkpoints/WGAN_GP_Conv1D', help='Carpeta donde guardar checkpoints')
    parser.add_argument('--L', type=int, default=128, help='Longitud de las señales (número de muestras)')

    args = parser.parse_args()

    # dispositivo (GPU si está disponible)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    ### Cargar dataset ###
    data = np.load(args.data).astype(np.float32)

    # convertir a tensores de PyTorch y crear un DataLoader
    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Dataset cargado: {len(dataset)} señales de longitud {args.L}")

    # Crear instancias de ambos modelos
    G = Generator(args.z_dim, args.L).to(device)
    C = Discriminator(args.L).to(device)
    G.apply(weights_init)
    C.apply(weights_init)

    train_gan(G,C, loader, device,
              epochs = args.epochs,
              lr=args.lr,
              λ_gp=args.lambda_gp,
              n_critic=args.n_critic,
              z_dim=args.z_dim,
              patience=args.patience,
              min_delta=args.min_delta,
              save_dir=args.save_dir)