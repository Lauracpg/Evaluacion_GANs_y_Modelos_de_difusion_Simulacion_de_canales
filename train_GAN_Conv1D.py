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
        # Proyección del vector de ruido z a un espacio más grande
        self.fc = nn.Linear(z_dim, 128 * self.init_len)
        self.net = nn.Sequential(
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # Salida final: 2 canales (parte real e imaginaria del canal)
            nn.ConvTranspose1d(32, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, z):
        # z: vector aleatorio (ruido): entrada del generador
        x = self.fc(z)
        x = x.view(z.size(0), 128, self.init_len)
        # genera señal de canal sintético
        return self.net(x)

# ----- DISCRIMINATOR Conv1D ----- #
class Discriminator(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        # Capa final: clasificación binaria (real vs falso)
        self.fc = nn.Linear(128 * (L // 8), 1)

    def forward(self, x):
        # x: señal real o generada
        f = self.conv(x)
        f = f.view(x.size(0), -1)
        return self.fc(f)

def train_gan(G, D, loader, device, epochs=100, lr=2e-4,
              save_dir='checkpoints/DCGAN_Conv1D', z_dim=32):

    os.makedirs(save_dir, exist_ok=True)
    # Función de pérdida binaria (real vs falso)
    criterion = nn.BCEWithLogitsLoss()
    # Optimizadores independientes para G y D
    optD = torch.optim.Adam(D.parameters(), lr=lr)
    optG = torch.optim.Adam(G.parameters(), lr=lr)

    # Mejor pérdida del generador (para guardar mejor modelo)
    best_g_loss = float("inf")

    ### Bucle del ENTRENAMIENTO principal ###
    # en cada iter:
    #   - entrenar primero discriminador (D)
    #   - luego el generador (G)
    for epoch in range(epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        for (real_batch,) in loader:
            real = real_batch.to(device)
            bsize = real.size(0)

            ### 1) TRAIN DISCRIMINADOR ###
            optD.zero_grad()
            # Generar muestras falsas
            z = torch.randn(bsize, z_dim, device=device)
            fake = G(z).detach()
            # detach() evita actualizar el generador en esta fase

            # Predicciones del discriminador
            pred_real = D(real) # debe clasificar como 1
            pred_fake = D(fake) # debe clasificar como 0

            # Etiquetas reales y falsas
            real_labels = torch.ones(bsize, 1, device=device)
            fake_labels = torch.zeros(bsize, 1, device=device)

            # Pérdida del discriminador:
            # - aprender a distinguir real vs falso
            loss_real = criterion(pred_real, real_labels)
            loss_fake = criterion(pred_fake, fake_labels)

            lossD = loss_real + loss_fake
            lossD.backward()
            optD.step()

            ### 2) TRAIN GENERADOR ###
            optG.zero_grad()

            # Generar nuevas muestras falsas
            z = torch.randn(bsize, z_dim, device=device)
            fake = G(z)

            # El generador intenta engañar al discriminador:
            # quiere que D(fake) ≈ 1 (parezca real)
            pred_fake = D(fake)
            lossG = criterion(pred_fake, real_labels)

            lossG.backward()
            optG.step()

            # Acumular pérdidas
            g_loss_epoch += lossG.item()
            d_loss_epoch += lossD.item()

        print(f"Epoch {epoch+1}/{epochs} | G: {g_loss_epoch:.4f} | D: {d_loss_epoch:.4f}")

        # Guardar mejor modelo (según pérdida del generador)
        if g_loss_epoch < best_g_loss:
            best_g_loss = g_loss_epoch

            torch.save({
                "G": G.state_dict(),
                "D": D.state_dict(),
                "epoch": epoch,
                "g_loss": g_loss_epoch,
                "d_loss": d_loss_epoch
            }, os.path.join(save_dir, "best_model.pth"))

            print(f"Nuevo mejor modelo guardado (epoch {epoch + 1})")

    print("Entrenamiento completado. Modelos guardados en", save_dir)

if __name__ == "__main__":
    ### Parámetros de ejecución ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/datos_sinteticos/dataset_synthetic.npy', help='Ruta del dataset .npy')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--z_dim', type=int, default=32, help='Dimensión del vector de ruido del generador')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (tasa de aprendizaje)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/GAN_Conv1D',
                        help='Carpeta donde guardar checkpoints')
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
    D = Discriminator(args.L).to(device)

    train_gan(G,D, loader, device,
              epochs = args.epochs,
              lr=args.lr,
              save_dir=args.save_dir,
              z_dim=args.z_dim)