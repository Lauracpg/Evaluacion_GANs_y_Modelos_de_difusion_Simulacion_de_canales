import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from scipy.signal import welch

from train_Diffusion_Model import UNet1D, DDPM
from train_GAN_Conv1D import Generator

def compute_energy(signals):
    """Energía total de cada señal"""
    return np.sum(np.abs(signals)**2, axis=1)

def compute_std_per_tap(signals):
    """Desviación estándar por tap (columnas)"""
    return np.std(signals, axis=0)

def compute_autocorrelation(signals):
    """Autocorrelación promedio por señal"""
    N, L = signals.shape
    autocorrelation = np.zeros(L)
    for s in signals:
        s = s - np.mean(s)
        ac = np.correlate(s, s, mode='full')
        ac = ac[ac.size//2:] # solo la mitad positiva
        autocorrelation += ac / (np.max(ac) + 1e-12)
    return autocorrelation / N

def compute_histogram(signals, bins=50, range=(-1,1)):
    """Histograma normalizado de magnitudes |h[n]|"""
    mags = np.abs(signals).flatten()
    hist, edges = np.histogram(mags, bins=bins, range=range, density=True)
    return hist, edges[:-1]

def compute_pdp(signals):
    power = np.abs(signals) ** 2
    return np.mean(power, axis=0)

def compute_average_delay(signals):
    N, L = signals.shape
    delays = np.arange(L)
    power = np.abs(signals) ** 2
    num = np.sum(delays * power)
    den = np.sum(power) + 1e-12
    return num / den

def compute_rms_delay_spread(signals):
    N, L = signals.shape
    delays = np.arange(L)
    power = np.abs(signals) ** 2
    avg_delay = compute_average_delay(signals)
    num = np.sum(power * (delays - avg_delay) ** 2)
    den = np.sum(power) + 1e-12
    return np.sqrt(num / den)

def compute_psd(signals, fs=1.0):
    """PSD promedio de las señales"""
    psds = []
    for s in signals:
        f, Pxx = welch(s, fs=fs, nperseg=min(128,len(s)))
        psds.append(Pxx)
    psds = np.array(psds)
    return np.mean(psds, axis=0), f

def mae(a, b):
    return np.mean(np.abs(a - b))

def evaluate_metrics(real, fake):
    """Calcula todas las métricas y devuelve un diccionario con resultados y datos para gráficas."""
    metrics = {}

    # Métricas clásicas
    metrics['pdp_real'] = compute_pdp(real)
    metrics['pdp_fake'] = compute_pdp(fake)
    metrics['pdp_mae'] = mae(metrics['pdp_real'], metrics['pdp_fake'])

    metrics['avg_real'] = compute_average_delay(real)
    metrics['avg_fake'] = compute_average_delay(fake)
    metrics['avg_delay_mae'] = abs(metrics['avg_real'] - metrics['avg_fake'])

    metrics['rms_real'] = compute_rms_delay_spread(real)
    metrics['rms_fake'] = compute_rms_delay_spread(fake)
    metrics['rms_mae'] = abs(metrics['rms_real'] - metrics['rms_fake'])

    # Métricas adicionales
    metrics['energy_real'] = compute_energy(real)
    metrics['energy_fake'] = compute_energy(fake)
    metrics['energy_mae'] = mae(metrics['energy_real'], metrics['energy_fake'])

    metrics['std_real'] = compute_std_per_tap(real)
    metrics['std_fake'] = compute_std_per_tap(fake)
    metrics['std_mae'] = mae(metrics['std_real'], metrics['std_fake'])

    metrics['autocorr_real'] = compute_autocorrelation(real)
    metrics['autocorr_fake'] = compute_autocorrelation(fake)
    metrics['autocorr_mae'] = mae(metrics['autocorr_real'], metrics['autocorr_fake'])

    # Histogramas
    metrics['hist_real'], metrics['bins_real'] = compute_histogram(real)
    metrics['hist_fake'], metrics['bins_fake'] = compute_histogram(fake)

    # PSD
    metrics['psd_real'], metrics['freqs'] = compute_psd(real)
    metrics['psd_fake'], _ = compute_psd(fake)
    metrics['psd_mae'] = mae(metrics['psd_real'], metrics['psd_fake'])

    # Diversidad: std de energía por muestra
    metrics['diversity_fake'] = np.std(metrics['energy_fake'])
    metrics['diversity_real'] = np.std(metrics['energy_real'])
    return metrics

def generate_signals(model_type, model_checkpoint, num_samples,
                     device, z_dim=32, L=128, ddpm_steps=1000):
    """
    Genera señales sintéticas de canal utilizando un modelo ya entrenado,
    sin alterar los pesos ni la distribución aprendida.
    Para obtener señales falsas comparables con las reales.
    Return:
    array (num_samples, L) con las señales generadas (normalizadas en [-1,1]).
    """
    # DCGAN: Generación directa a partir de ruido latente z
    if model_type == 'dcgan':
        # cargar checkpoint y reconstruir el generador
        checkpoint = torch.load(model_checkpoint, map_location=device)
        G = Generator(z_dim=z_dim, L=L).to(device)
        G.load_state_dict(checkpoint['G'])
        G.eval()

        # Muestreo directo: x = G(z)
        with torch.no_grad():
            z = torch.randn(num_samples, z_dim, device=device)
            fake = G(z).squeeze(1).cpu().numpy()
    # DDPM: generación mediante proceso inverso de difusión
    elif model_type == 'ddpm':
        # cargar U-Net entrenada
        checkpoint = torch.load(model_checkpoint, map_location=device)
        unet = UNet1D(L=L).to(device)
        unet.load_state_dict(checkpoint)

        # reconstruir DDPM con el mismo número de pasos T
        ddpm = DDPM(unet, T=ddpm_steps).to(device)
        ddpm.model.eval()

        fake = []

        with torch.no_grad():
            for _ in range(num_samples):
                # Inicialización desde ruido gaussiano puro
                x = torch.randn(1, 1, L, device=device)

                # Proceso inverso de difusión: eliminar ruido paso a paso
                # desde t=T-1 hasta t=0
                for t in reversed(range(ddpm.T)):
                    noise_pred = ddpm.model(x,torch.tensor([t], device=device))
                    beta = ddpm.betas[t]
                    # ecuación de muestreo DDPM (simplificada)
                    x = (1 / torch.sqrt(1-beta) * (
                        x - beta / torch.sqrt(1 - ddpm.alphas_cumprod[t]) * noise_pred)
                    )
                # guardar señal final x_0
                x0 = x.squeeze()
                x0= torch.clamp(x0, -1, 1) # [-1,1]
                fake.append(x0.cpu().numpy())
        fake = np.array(fake)
    else:
        raise ValueError('Invalid model type, debe ser "dcgan" o "ddpm"')

    return fake

# MAIN
def main(args):
    print("\n=== Cargando datos reales ===")
    data = np.load(args.data).astype(np.float32)
    real_eval = data[:args.num_eval]  # primeras N señales
    print("Dataset cargado:", data.shape)

    print(f"\n=== Generando señales con {args.model_type.upper()} ===")
    fake_eval = generate_signals(
        args.model_type,args.model, args.num_eval,args.device,
        z_dim=args.z_dim, L=args.L,ddpm_steps=args.ddpm_steps
    )
    print("Señales generadas:", fake_eval.shape)

    # calcular métricas
    metrics = evaluate_metrics(real_eval, fake_eval)

    print("\n--- RESULTADOS ---")
    for key in ['pdp_mae', 'avg_delay_mae', 'rms_mae','energy_mae',
                'std_mae', 'autocorr_mae', 'psd_mae']:
        print(f"{key}: {metrics[key]:.6f}")
    print(f"Diversidad fake: {metrics['diversity_fake']:.6f}")

    # Guardar resultados
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, 'metrics.npy'), metrics)

    # Gráfico PDP
    plt.figure(figsize=(8, 4))
    plt.plot(metrics['pdp_real'], label="Real")
    plt.plot(metrics['pdp_fake'], label="Generado")
    plt.title("PDP Real vs Generado")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'pdp.png'))

    # Histograma de magnitudes
    plt.figure(figsize=(8, 4))
    plt.bar(metrics['bins_real'],metrics['hist_real'],width=0.03, alpha=0.6, label="Real")
    plt.bar(metrics['bins_fake'], metrics['hist_fake'], width=0.03, alpha=0.6, label="Generado")
    plt.title("Histograma de magnitudes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'histograma.png'))

    # Gráfico PSD
    plt.figure(figsize=(8, 4))
    plt.semilogy(metrics['freqs'], metrics['psd_real'], label='Real')
    plt.semilogy(metrics['freqs'],metrics['psd_fake'], label='Generado')
    plt.title("PSD Real vs Generado")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "psd.png"))

    # Señales reales vs generadas (8)
    num_plot = min(8, args.num_eval)
    plt.figure(figsize=(12, 6))
    for i in range(num_plot):
        plt.subplot(4,2,i+1)
        plt.plot(real_eval[i], label='Real')
        plt.plot(fake_eval[i], '--',label='Generado')
        plt.xticks([]); plt.yticks([])
    plt.suptitle(f"Comparación señales reales vs {args.model_type.upper()}")
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "real_vs_generada.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dataset_nist.npy")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=['dcgan', 'ddpm'], default='dcgan')
    parser.add_argument("--save_dir", type=str, default="eval_results")
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--num_eval", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ddpm_steps", type=int, default=1000)
    args = parser.parse_args()
    main(args)