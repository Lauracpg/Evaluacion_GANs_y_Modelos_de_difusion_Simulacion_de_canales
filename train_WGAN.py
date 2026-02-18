import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.data import TensorDataset, DataLoader

# ----- GENERATOR Conv1D ----- #
class Generator(nn.Module):
    def __init__(self, z_dim, L):
        super().__init__()
        self.init_len = L // 8
        self.fc = nn.Linear(z_dim, 128 * self.init_len)
        self.net = nn.Sequential(
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, self.init_len)
        return self.net(x)


# ----- DISCRIMINATOR Conv1D ----- #
class Discriminator(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv1d(1, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv1d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv1d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
        )
        self.fc = spectral_norm(nn.Linear(128 * (L // 8), 1))

    def forward(self, x):
        f = self.net(x)
        f = f.view(x.size(0), -1)
        return self.fc(f)

### Inicialización DCGAN ###
def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def gradient_penalty(D, real, fake, device):
    bsize = real.size(0)
    epsilon = torch.rand(bsize, 1, 1, device=device)
    epsilon = epsilon.expand_as(real)

    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    d_interpolated = D(interpolated)

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

def train_gan(G, D, loader, device, epochs=100, lr=2e-4,
              save_dir='checkpoints/GAN_Conv1D',patience=10,
              min_delta=1e-4, z_dim=32):
    os.makedirs(save_dir, exist_ok=True)
    # pérdida LSGAN
    #criterion = nn.MSELoss()
    # optimizadores Adam para G y D
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    ### Bucle del ENTRENAMIENTO principal ###
    # en cada iter:
    #   - entrenar primero discriminador (D)
    #   - luego el generador (G)
    #   - guardar las pérdidas promedio por época

    λ_gp = 10
    λ_pdp = 0.1
    n_critic = 5

    best_g_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        g_loss_avg = 0.0
        d_loss_avg = 0.0

        for (real_batch,) in loader:
            real = real_batch.to(device)
            real = real.unsqueeze(1)
            bsize = real.size(0)

            d_loss_batch = 0.0

            ### Entrenar Discriminador ###
            for _ in range(n_critic):
                optD.zero_grad()
                z = torch.randn(bsize, z_dim, device=device)
                fake = G(z).detach()

                loss_D = -(D(real).mean() - D(fake).mean())
                gp = gradient_penalty(D, real, fake, device)

                loss_D_total = loss_D + λ_gp * gp
                loss_D_total.backward()
                optD.step()

                d_loss_batch += loss_D.item()

            ### Entrenar Generador ###
            optG.zero_grad()

            z = torch.randn(bsize, z_dim, device=device)
            fake = G(z)

            loss_adv = -D(fake).mean()

            pdp_real = compute_pdp(real).detach()
            pdp_fake = compute_pdp(fake)
            loss_pdp = torch.mean(torch.abs(pdp_real - pdp_fake))

            loss_G_total = loss_adv + λ_pdp * loss_pdp
            loss_G_total.backward()
            optG.step()

            # acumular pérdidas
            g_loss_avg += loss_G_total.item()
            d_loss_avg += d_loss_batch / n_critic

        # promedios por época
        g_loss_avg /= len(loader)
        d_loss_avg /= len(loader)

        # mostrar cada 20 épocas
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | G_loss={g_loss_avg:.4f} | D_loss={d_loss_avg:.4f}")

        # Guardar mejor modelo (según menor G_loss)
        if g_loss_avg < best_g_loss - min_delta:
            best_g_loss = g_loss_avg
            epochs_no_improve = 0
            torch.save(
                {
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'epoch': epoch,
                    'G_loss': g_loss_avg,
                    'D_loss': d_loss_avg
                },
                os.path.join(save_dir, 'model_best.pth')
            )
            print(f"Nuevo mejor modelo en la época {epoch} | G_loss={g_loss_avg:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping en época {epoch} (sin mejora en {patience} épocas)")
            break

    print("Entrenamiento completado. Modelos guardados en", save_dir)
    print(f"Mejor G_loss: {best_g_loss:.4f}")

if __name__ == "__main__":
    ### Parámetros de ejecución ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dataset_nist.npy', help='Ruta del dataset .npy')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--z_dim', type=int, default=32, help='Dimensión del vector de ruido del generador')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (tasa de aprendizaje)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/WGAN_Conv1D',
                        help='Carpeta donde guardar checkpoints')
    parser.add_argument('--L', type=int, default=128, help='Longitud de las señales (número de muestras)')
    parser.add_argument('--patience', type=int, default=10, help='Paciencia para early stopping')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Mejora mínima en G_loss')

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
    D = Discriminator(args.L).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    train_gan(G,D, loader, device,
              epochs = args.epochs,
              lr=args.lr,
              save_dir=args.save_dir,
              patience=args.patience,
              min_delta=args.min_delta,
              z_dim=args.z_dim)