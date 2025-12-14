import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
#from train_GAN_fc import Generator
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

    return metrics

###############################################################
# ARCHIVO PRINCIPAL (EVALUACIÓN)
###############################################################

def main(args):
    print("\n=== Cargando datos reales ===")
    data = np.load(args.data).astype(np.float32)
    real_eval = data[:args.num_eval]  # primeras N señales
    print("Dataset cargado:", data.shape)

    print("\n=== Cargando modelo entrenado ===")
    checkpoint = torch.load(args.model, map_location=args.device)
    G_state = checkpoint["G"]

    # Cargar generador
    G = Generator(z_dim=args.z_dim, out_dim=args.L).to(args.device)
    G.load_state_dict(G_state)
    G.eval()
    print("Modelo cargado correctamente.")

    print("\n=== Generando señales falsas para evaluación ===")
    with torch.no_grad():
        z = torch.randn(args.num_eval, args.z_dim, device=args.device)
        fake_eval = G(z).cpu().numpy()
    print("Listo. Real vs Fake shapes:", real_eval.shape, fake_eval.shape)

    # Calcular métricas
    print("\n=== Calculando métricas ===")
    metrics = evaluate_metrics(real_eval, fake_eval)

    print("\n--- RESULTADOS ---")
    print(f"PDP MAE = {metrics['pdp_mae']:.6f}")
    print(f"Average Delay MAE = {metrics['avg_delay_mae']:.6f}")
    print(f"RMS Delay Spread MAE = {metrics['rms_mae']:.6f}")
    print(f"Energía total MAE = {metrics['energy_mae']:.6f}")
    print(f"Std por tap MAE = {metrics['std_mae']:.6f}")
    print(f"Autocorrelación MAE = {metrics['autocorr_mae']:.6f}")

    # Guardar resultados
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "metricas_gan.txt"), "w") as f:
        f.write("=== Métricas clásicas ===\n")
        f.write(f"PDP MAE: {metrics['pdp_mae']}\n")
        f.write(f"Average Delay MAE: {metrics['avg_delay_mae']}\n")
        f.write(f"RMS Delay Spread MAE: {metrics['rms_mae']}\n")
        f.write("\n=== Métricas adicionales ===\n")
        f.write(f"Energía total MAE: {metrics['energy_mae']}\n")
        f.write(f"Std por tap MAE: {metrics['std_mae']}\n")
        f.write(f"Autocorrelación MAE: {metrics['autocorr_mae']}\n")
    print("\nMétricas guardadas en metricas_gan.txt")

    # Gráfico PDP
    plt.figure(figsize=(8, 4))
    plt.plot(metrics['pdp_real'], label="Real")
    plt.plot(metrics['pdp_fake'], label="Generado")
    plt.title("PDP Real vs Generado")
    plt.xlabel("Delay (muestra)")
    plt.ylabel("Potencia promedio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "pdp_comparacion.png"))
    print("Gráfico PDP guardado en pdp_comparacion.png")

    # Histograma de magnitudes
    plt.figure(figsize=(8, 4))
    plt.bar(metrics['bins_real'], metrics['hist_real'], width=(metrics['bins_real'][1] - metrics['bins_real'][0]),
            alpha=0.5, label="Real")
    plt.bar(metrics['bins_fake'], metrics['hist_fake'], width=(metrics['bins_fake'][1] - metrics['bins_fake'][0]),
            alpha=0.5, label="Generado")
    plt.title("Distribución de magnitudes |h[n]|")
    plt.xlabel("Magnitud")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "histograma_magnitudes.png"))
    print("Histograma de magnitudes guardado en histograma_magnitudes.png")

###############################################################
# ARGUMENTOS
###############################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dataset_nist.npy")
    parser.add_argument("--model", type=str, default="checkpoints/fc/model_best.pth")
    parser.add_argument("--save_dir", type=str, default="eval_results")
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--num_eval", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)

