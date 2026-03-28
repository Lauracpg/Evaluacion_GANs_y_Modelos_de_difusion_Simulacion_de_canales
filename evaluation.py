import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import ttest_ind

from train_DDIM import Diffusion, DDIMSampler
from train_DDPM import UNet1D, DDPM
#from train_GAN_Conv1D import Generator
#from train_DCGAN_Conv1D import Generator
from train_WGAN import Generator

def to_db(x, eps=1e-10):
    return 10 * np.log10(np.maximum(x, eps))

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

def compute_average_delay(signals, delta_tau=1.0):
    """
    Average delay por señal y luego promedio global.
    """
    N, L = signals.shape
    delays = np.arange(L) * delta_tau
    avg_delays = []
    for s in signals:
        power = np.abs(s) ** 2
        total_power = np.sum(power) + 1e-12
        mean_delay = np.sum(delays * power) / total_power
        avg_delays.append(mean_delay)

    avg_delays = np.array(avg_delays)
    return np.mean(avg_delays), avg_delays

def compute_rms_delay_spread(signals, delta_tau=1.0):
    """
    RMS delay spread por señal y luego promedio global.
    """
    N, L = signals.shape
    delays = np.arange(L) * delta_tau
    rms_values = []
    for s in signals:
        power = np.abs(s) ** 2
        total_power = np.sum(power) + 1e-12
        mean_delay = np.sum(delays * power) / total_power
        rms = np.sqrt(np.sum(power * (delays - mean_delay) ** 2) / total_power)
        rms_values.append(rms)

    rms_values = np.array(rms_values)
    return np.mean(rms_values), rms_values

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

def evaluate_metrics(real, fake, delta_tau=1.0):
    """Calcula todas las métricas y devuelve un diccionario con resultados y datos para gráficas."""
    metrics = {}

    # Métricas clásicas
    metrics['pdp_real'] = compute_pdp(real)
    metrics['pdp_fake'] = compute_pdp(fake)
    metrics['pdp_mae'] = mae(metrics['pdp_real'], metrics['pdp_fake'])
    metrics['pdp_real_db'] = to_db(metrics['pdp_real'])
    metrics['pdp_fake_db'] = to_db(metrics['pdp_fake'])

    mask = metrics['pdp_real'] > 1e-6  # o equivalente en potencia
    metrics['pdp_mae_db'] = np.mean(
        np.abs(metrics['pdp_real_db'][mask] - metrics['pdp_fake_db'][mask])
    )

    # Average delay por señal
    metrics['avg_real'], metrics['avg_real_all'] = compute_average_delay(real, delta_tau)
    metrics['avg_fake'], metrics['avg_fake_all'] = compute_average_delay(fake, delta_tau)
    metrics['avg_delay_mae'] = abs(metrics['avg_real'] - metrics['avg_fake'])
    metrics['avg_delay_mae_ns'] = metrics['avg_delay_mae'] * 1e9

    # RMS delay spread por señal
    metrics['rms_real'], metrics['rms_real_all'] = compute_rms_delay_spread(real, delta_tau)
    metrics['rms_fake'], metrics['rms_fake_all'] = compute_rms_delay_spread(fake, delta_tau)
    metrics['rms_mae'] = abs(metrics['rms_real'] - metrics['rms_fake'])
    metrics['rms_mae_ns'] = metrics['rms_mae'] * 1e9
    # T-Test Student
    t_stat, t_pvalue = ttest_ind(
        metrics['rms_real_all'],
        metrics['rms_fake_all'],
        equal_var=False
    )
    metrics['rms_t_stat'] = t_stat
    metrics['rms_t_pvalue'] = t_pvalue

    t_stat, t_pvalue = ttest_ind(
        metrics['avg_real_all'],
        metrics['avg_fake_all'],
        equal_var=False
    )
    metrics['avg_t_stat'] = t_stat
    metrics['avg_t_pvalue'] = t_pvalue

    # Métricas adicionales
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

    metrics['psd_real_db'] = to_db(metrics['psd_real'])
    metrics['psd_fake_db'] = to_db(metrics['psd_fake'])
    metrics['psd_mae_db'] = mae(metrics['psd_real_db'], metrics['psd_fake_db'])

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

    # WGAN-GP
    elif model_type == 'wgan':
        checkpoint = torch.load(model_checkpoint, map_location=device)
        G = Generator(z_dim=z_dim, L=L).to(device)
        G.load_state_dict(checkpoint['G'])
        G.eval()

        with torch.no_grad():
            z = torch.randn(num_samples, z_dim, device=device)
            fake = G(z).cpu().numpy()

    # DDPM: generación mediante proceso inverso de difusión
    elif model_type == 'ddpm':
        # cargar U-Net entrenada
        checkpoint = torch.load(model_checkpoint, map_location=device)
        unet = UNet1D(L=L).to(device)
        unet.load_state_dict(checkpoint)

        # reconstruir DDPM con el mismo número de pasos T
        ddpm = DDPM(unet, T=ddpm_steps).to(device)
        ddpm.model.eval()

        with torch.no_grad():
            # Inicialización desde ruido gaussiano puro
            x = torch.randn(num_samples, 2, L, device=device)

            # Proceso inverso de difusión: eliminar ruido paso a paso
            # desde t=T-1 hasta t=0
            for t in reversed(range(ddpm.T)):
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
                noise_pred = ddpm.model(x, t_batch)
                beta = ddpm.betas[t]

                if t > 0:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)

                # ecuación de muestreo DDPM (simplificada)
                x = (
                        (1 / torch.sqrt(1 - beta)) *
                        (x - beta / torch.sqrt(1 - ddpm.alphas_cumprod[t]) * noise_pred)
                        + torch.sqrt(beta) * z
                )
            # guardar señal final x_0
            x0 = x
            # energy = torch.sqrt(torch.sum(x0 ** 2, dim=2, keepdim=True))
            # x0 = x0 / (energy + 1e-12)
            x0 = torch.clamp(x0, -1, 1)
            fake = x0.cpu().numpy()
    elif model_type == 'ddim':
        checkpoint = torch.load(model_checkpoint, map_location=device)

        unet = UNet1D().to(device)
        unet.load_state_dict(checkpoint)
        unet.eval()

        diffusion = Diffusion(unet, T=ddpm_steps).to(device)
        sampler = DDIMSampler(diffusion, eta=0.0)  # determinista

        with torch.no_grad():
            fake = sampler.sample(
                shape=(num_samples, 2, L),
                device=device,
                steps=50   # menos pasos que DDPM
            ).cpu().numpy()
    else:
        raise ValueError('Invalid model type, debe ser "dcgan" o "ddpm"')

    return fake

# MAIN
def main(args):
    print("\n=== Cargando datos reales ===")
    np.random.seed(42)
    torch.manual_seed(42)
    data = np.load(args.data).astype(np.float32)
    idx = np.random.choice(len(data), args.num_eval, replace=False)
    real_eval = data[idx]
    print("Dataset cargado:", data.shape)

    print(f"\n=== Generando señales con {args.model_type.upper()} ===")
    fake_eval = generate_signals(
        args.model_type, args.model, args.num_eval, args.device,
        z_dim=args.z_dim, L=args.L, ddpm_steps=args.ddpm_steps
    )
    print("Señales generadas:", fake_eval.shape)

    real_eval_complex = real_eval[:, 0, :] + 1j * real_eval[:, 1, :]
    fake_eval_complex = fake_eval[:, 0, :] + 1j * fake_eval[:, 1, :]
    real_mag = np.abs(real_eval_complex)
    fake_mag = np.abs(fake_eval_complex)

    # calcular métricas
    metrics = evaluate_metrics(real_mag, fake_mag, delta_tau=args.delta_tau)

    print("\n--- RESULTADOS ---")
    for key in ['pdp_mae', 'pdp_mae_db',
                'avg_delay_mae_ns', 'rms_mae_ns',
                'std_mae', 'autocorr_mae',
                'psd_mae', 'psd_mae_db']:
        if key in ['avg_delay_mae_ns', 'rms_mae_ns']:
            print(f"{key}: {metrics[key]:.3f} ns")

        elif key == 'pdp_mae_db':
            print(f"{key}: {metrics[key]:.3f} dB")

        elif key == 'psd_mae_db':
            print(f"{key}: {metrics[key]:.3f} dB/Hz")

        else:
            print(f"{key}: {metrics[key]:.6f}")

    ns = 1e9

    print("\n--- MÉTRICAS REALES ---")
    print(f"Avg delay real: {metrics['avg_real'] * ns:.3f} ns")
    print(f"RMS delay real: {metrics['rms_real'] * ns:.3f} ns")
    print(f"Mean PSD real: {np.mean(metrics['psd_real_db']):.3f} dB/Hz")

    print("\n--- MÉTRICAS GENERADAS ---")
    print(f"Avg delay fake: {metrics['avg_fake'] * ns:.3f} ns")
    print(f"RMS delay fake: {metrics['rms_fake'] * ns:.3f} ns")
    print(f"Mean PSD fake: {np.mean(metrics['psd_fake_db']):.3f} dB/Hz")

    # Guardar resultados
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(os.path.join(args.save_dir, 'metrics.npy'), metrics)

    scalar_keys = [
        'pdp_mae',
        'pdp_mae_db',
        'avg_delay_mae_ns',
        'rms_mae_ns',
        'std_mae',
        'autocorr_mae',
        'psd_mae',
        'psd_mae_db',
    ]

    txt_path = os.path.join(args.save_dir, "metrics.txt")

    def significado_ttest(p):
        if p > 0.05:
            return "STATISTICALLY SIMILAR"
        else:
            return "STATISTICALLY DIFFERENT"

    with open(txt_path, "w") as f:
        f.write("===== RESULTADOS EVALUACION =====\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Checkpoint: {args.model}\n")
        f.write(f"Num eval samples: {args.num_eval}\n")
        f.write("=" * 50 + "\n\n")

        # Métricas globales comparación
        f.write("---- METRICAS GLOBALES ----\n")
        for k in scalar_keys:
            if k in ['avg_delay_mae_ns', 'rms_mae_ns']:
                f.write(f"{k}: {metrics[k]:.3f} ns\n")
            elif k == 'pdp_mae_db':
                f.write(f"{k}: {metrics[k]:.3f} dB\n")
            elif k == 'psd_mae_db':
                f.write(f"{k}: {metrics[k]:.3f} dB/Hz\n")
            else:
                f.write(f"{k}: {metrics[k]:.6f}\n")

        f.write("\n---- STATISTICAL TESTS ----\n")
        f.write(f"RMS t-test statistic: {metrics['rms_t_stat']:.6f}\n")
        f.write(f"RMS t-test p_value: {metrics['rms_t_pvalue']:.3e}\n")
        f.write(f"AVG delay t-test statistic: {metrics['avg_t_stat']:.6f}\n")
        f.write(f"AVG delay t-test p_value: {metrics['avg_t_pvalue']:.3e}\n")

        f.write("\n")

        # Estadísticas REALES
        f.write("---- REAL CHANNEL STATS ----\n")
        f.write(f"Avg delay real: {metrics['avg_real'] * 1e9:.3f} ns\n")
        f.write(f"RMS delay real: {metrics['rms_real'] * 1e9:.3f} ns\n")
        f.write(f"Mean PSD real: {np.mean(metrics['psd_real_db']):.3f} dB/Hz\n")

        f.write("\n")

        # Estadísticas GENERADAS
        f.write("---- GENERATED CHANNEL STATS ----\n")
        f.write(f"Avg delay fake: {metrics['avg_fake'] * 1e9:.3f} ns\n")
        f.write(f"RMS delay fake: {metrics['rms_fake'] * 1e9:.3f} ns\n")
        f.write(f"Mean PSD fake: {np.mean(metrics['psd_fake_db']):.3f} dB/Hz\n")
        f.write("\n")

        f.write("\n---- STATISTICAL INTERPRETATION ----\n")
        f.write(
            f"RMS comparison: "
            f"{significado_ttest(metrics['rms_t_pvalue'])}\n"
        )
        f.write(
            f"AVG delay comparison: "
            f"{significado_ttest(metrics['avg_t_pvalue'])}\n"
        )
    # Gráfico PDP
    plt.figure(figsize=(8, 4))
    delays = np.arange(len(metrics['pdp_real_db'])) * args.delta_tau * 1e9  # ns
    plt.plot(delays, metrics['pdp_real_db'], label="Real")
    plt.plot(delays, metrics['pdp_fake_db'], label="Generado")
    plt.title("PDP (dB) Real vs Generado")
    plt.xlabel("Delay (ns)")
    plt.ylabel("Power (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'pdp.png'))

    # Histograma de magnitudes
    plt.figure(figsize=(8, 4))
    plt.bar(metrics['bins_real'], metrics['hist_real'], width=0.03, alpha=0.6, label="Real")
    plt.bar(metrics['bins_fake'], metrics['hist_fake'], width=0.03, alpha=0.6, label="Generado")
    plt.title("Histograma de magnitudes")
    plt.xlabel("|h[n]|")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'histograma.png'))

    # Gráfico PSD
    plt.figure(figsize=(8, 4))
    plt.plot(metrics['freqs'], metrics['psd_real_db'], label='Real')
    plt.plot(metrics['freqs'], metrics['psd_fake_db'], label='Generado')
    plt.title("PSD (db/Hz) Real vs Generado")
    plt.xlabel("Frequency (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "psd.png"))

    # Señales reales vs generadas (8)
    num_plot = min(8, args.num_eval)
    plt.figure(figsize=(12, 6))
    for i in range(num_plot):
        plt.subplot(4, 2, i + 1)
        plt.plot(real_mag[i], label='Real')
        plt.plot(fake_mag[i], '--', label='Generado')
        plt.xticks([]);
        plt.yticks([])
    plt.suptitle(f"Comparación señales reales vs {args.model_type.upper()}")
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "real_vs_generada.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/datos_sinteticos/dataset_synthetic.npy")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=['dcgan', 'wgan', 'ddpm', 'ddim'], default='dcgan')
    parser.add_argument("--save_dir", type=str, default="eval_results")
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--num_eval", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ddpm_steps", type=int, default=1000)
    parser.add_argument("--delta_tau", type=float, default=1e-9)  # segundos (ej: 1 ns)
    args = parser.parse_args()

    base_save_dir = "eval_results"
    if args.model_type == "dcgan":
        args.save_dir = os.path.join(base_save_dir, "DCGAN_Conv1D")

    elif args.model_type == "wgan":
        args.save_dir = os.path.join(base_save_dir, "WGAN_GP_Conv1D")

    elif args.model_type == "ddpm":
        args.save_dir = os.path.join(base_save_dir, "DDPM_Conv1D")

    elif args.model_type == "ddim":
        args.save_dir = os.path.join(base_save_dir, "DDIM_Conv1D")

    else:
        raise ValueError("model_type inválido")

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)