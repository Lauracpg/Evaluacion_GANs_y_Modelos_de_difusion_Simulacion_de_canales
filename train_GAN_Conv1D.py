import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

### Parámetros de ejecución ###
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/dataset_nist.npy', help='Ruta del dataset .npy')
parser.add_argument('--epochs', type=int, default=100, help='Número de épocas de entrenamiento')
parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
parser.add_argument('--z_dim', type=int, default=32, help='Dimensión del vector de ruido del generador')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (tasa de aprendizaje)')
parser.add_argument('--save_dir', type=str, default='checkpoints/fc', help='Carpeta donde guardar checkpoints')
parser.add_argument('--L', type=int, default=128, help='Longitud de las señales (número de muestras)')
args = parser.parse_args()

# dispositivo (GPU si está disponible)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(args.save_dir, exist_ok=True)

### Cargar dataset ###
data = np.load(args.data).astype(np.float32)
data = data / (np.max(np.abs(data), axis=1, keepdims=True) + 1e-12)
# normalizar a [-1,1]
data = 2*data - 1

# convertir a tensores de PyTorch y crear un DataLoader
dataset = TensorDataset(torch.from_numpy(data))
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
print(f"Dataset cargado: {len(dataset)} señales de longitud {args.L}")

### Definición de los modelos: Generator y Discriminator ###
def gen_block(in_f, out_f):
    # Cada bloque del Generado: Linear + BatchNorm + ReLU
    return [nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f), nn.ReLU(True)]

def dis_block(in_f, out_f):
    # Cada bloque del Discriminador: Linear + LeakyReLU + Dropout
    return [nn.Linear(in_f, out_f), nn.LeakyReLU(0.2, True), nn.Dropout(0.2)]

# ----- GENERATOR Conv1D ----- #
class Generator(nn.Module):
    def __init__(self, z_dim, L):
        super().__init__()
        self.init_len = L // 8
        self.fc = nn.Linear(z_dim, 128 * self.init_len)
        self.net = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, self.init_len)
        return self.net(x).squeeze(1)

# ----- DISCRIMINATOR Conv1D ----- #
class Discriminator(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(128 * (L // 8), 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        f = self.net(x)
        f = f.view(x.size(0), -1)
        return self.fc(f)

### Inicialización DCGAN ###
def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Crear instancias de ambos modelos
G = Generator(args.z_dim, args.L).to(device)
D = Discriminator(args.L).to(device)

G.apply(weights_init)
D.apply(weights_init)

### Definir pérdidas y optimizadores de G y D (LSGAN) ###
# pérdida LSGAN
criterion = nn.MSELoss()

# optimizadores Adam para G y D
optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

# vector de ruido fijo para ver la evolución del generador
fixed_z = torch.randn(16, args.z_dim, device=device)

### Bucle del ENTRENAMIENTO principal ###
# en cada iter:
#   - entrenar primero discriminador (D)
#   - luego el generador (G)
#   - guardar las pérdidas promedio por época
for epoch in range(1, args.epochs + 1):
    g_loss_avg = 0.0
    d_loss_avg = 0.0

    for(real_batch,) in loader:
        real = real_batch.to(device)
        bsize = real.size(0)

        ### Entrenar Discriminador ###
        optD.zero_grad()
        # ruido leve en señales reales (ruido de medición)
        real_noisy = real + 0.02 * torch.randn_like(real)
        # etiquetas suavizadas
        real_labels = torch.full((bsize, 1), 0.9, device=device)
        fake_labels = torch.full((bsize, 1), 0.1, device=device)
        # 1. Señales reales
        pred_real = D(real_noisy)
        loss_real = criterion(pred_real, real_labels)
        # 2. Señales falsas generadas
        z = torch.randn(bsize, args.z_dim, device=device)
        fake = G(z).detach()
        pred_fake = D(fake)
        loss_fake = criterion(pred_fake, fake_labels)
        # 3. Pérdida total del discriminador (real=1, fake=0)
        lossD = 0.5 * (loss_real + loss_fake)
        lossD.backward()
        optD.step()

        ### Entrenar Generador ###
        for _ in range(2):
            optG.zero_grad()
            z = torch.randn(bsize, args.z_dim, device=device)
            fake = G(z)
            pred_fake = D(fake)
            lossG = criterion(pred_fake, real_labels)
            lossG.backward()
            optG.step()

        # acumular pérdidas
        g_loss_avg += lossG.item()
        d_loss_avg += lossD.item()

    # promedios por época
    g_loss_avg /= len(loader)
    d_loss_avg /= len(loader)

    # mostrar cada 20 épocas
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{args.epochs} | G_loss={g_loss_avg:.4f} | D_loss={d_loss_avg:.4f}")
        # guardar checkpoint con pesos de los modelos
        torch.save(
            {'G': G.state_dict(), 'D': D.state_dict(), 'epoch': epoch,
             'G_loss': g_loss_avg, 'D_loss': d_loss_avg},
            os.path.join(args.save_dir, f'model_e{epoch}.pth')
        )

    # Guardar mejor modelo (según menor G_loss)
    if 0.4 < d_loss_avg < 0.8:
        torch.save(
            {'G': G.state_dict(), 'D': D.state_dict(),
             'epoch': epoch, 'G_loss': g_loss_avg, 'D_loss': d_loss_avg},
            os.path.join(args.save_dir, 'model_stable.pth')
        )

### Guardar modelo final y muestras generadas ###
# guardar pesos finales
torch.save(
{'G': G.state_dict(), 'D': D.state_dict(), 'epoch': args.epochs},
    os.path.join(args.save_dir, 'model_final.pth')
)

# guardar ejemplos generados con ruido fijo
G.eval()
with torch.no_grad():
    samples = G(fixed_z).cpu().numpy()

np.save(os.path.join(args.save_dir, 'fixed_samples.npy'), samples)
print("Entrenamiento completado. Modelos guardados en", args.save_dir)

# Visualizar algunas señales generadas
plt.figure(figsize=(10, 6))
for i in range(8):
    plt.subplot(4, 2, i+1)
    plt.plot(samples[i], '--', color='orange', alpha=0.8, label='Generada')
    plt.title(f'Señal generada {i+1}')
    plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.show()

print("\nVisualizando lo aprendido por el modelo final...")

# Generar nuevas señales
z_vis = torch.randn(8, args.z_dim, device=device)
with torch.no_grad():
    generated = G(z_vis).cpu().numpy()

# Seleccionar señales reales para comparar
num_examples = 8
real_examples = data[:num_examples]

# Comparar gráficamente modelo final
plt.figure(figsize=(12, 8))
for i in range(num_examples):
    plt.subplot(4, 2, i + 1)
    plt.plot(real_examples[i], color='blue', alpha=0.7, label='Real')
    plt.plot(generated[i], color='orange', linestyle='--', alpha=0.8, label='Generada')
    plt.title(f"Comparación señal {i + 1}")
    plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.legend(loc='upper right', fontsize=8)
plt.suptitle("Comparación de señales reales vs generadas (modelo final)", fontsize=14, y=1.02)

plt.savefig(os.path.join(args.save_dir, 'comparacion_real_vs_generada.png'))
plt.show()

# modelo mejor guardado
checkpoint = torch.load('checkpoints/fc/model_best.pth', map_location='cpu')
print(f"Modelo guardado en la época: {checkpoint['epoch']}")
print(f"Pérdida del generador (G_loss): {checkpoint['G_loss']:.4f}")
print(f"Pérdida del discriminador (D_loss): {checkpoint['D_loss']:.4f}")