import numpy as np
import os
import h5py
from matplotlib import pyplot as plt

def load_nist_data(mat_path, L=128, save_path="data/dataset_nist.npy"):
    print(f"Cargando archivo {mat_path} ...")
    f = h5py.File(mat_path, "r")

    print("Claves encontradas:")
    keys = list(f.keys())
    for k in keys:
        print(" ", k)

    # extraer todas las matrices que empiezan por "h_"
    channel_keys = [k for k in keys if k.startswith("h_")]
    print("\nMatrices de canal encontradas:")
    for ck in channel_keys:
        print(" ", ck)

    all_channels = []
    total_channels = 0

    # Procesar cada matriz h_XXXX
    for ck in channel_keys:
        print("\n" + "="*60)
        print(f"\nProcesando {ck} ...")
        print("="*60)

        H_raw = f[ck] # matriz 241 x YYY
        # conservar el dtype estructurado
        H_struct = H_raw[...]

        print("dtype de la matriz H:", H_struct.dtype)

        # reconstruir números complejos
        H_real = np.nan_to_num(H_struct["real"], nan=0.0)
        H_imag = np.nan_to_num(H_struct["imag"], nan=0.0)
        H = H_real + 1j * H_imag

        print("Shape original:", H.shape)

        # H es 241 x num_canales, a veces se invierte dependiendo de h5py
        if H.shape[0] != 241 and H.shape[1] == 241:
            print("La matriz está transpuesta. Corrigiendo orientación...")
            H = H.T
            print("Transpuesta:", H.shape)

        # H es 241 x YYY, YYY = número de canales
        print("Shape actual (debería ser 241 x N):", H.shape)

        # recortar a 128 muestras
        H_crop = H[:L, :] # shape: L x num_canales
        print("Shape tras recorte a L=128:", H_crop.shape)

        # convertir columnas -> filas
        H_crop = H_crop.T
        print(" Shape (canales x L):", H_crop.shape)

        num_c = H_crop.shape[0]
        total_channels += num_c
        print(f"Añadiendo {num_c} canales de tamaño {L}")

        # convertir a magnitud
        H_mag = np.abs(H_crop)

        # normalizar cada canal a [-1,1]
        max_vals = np.max(H_mag, axis=1, keepdims=True)
        print("Primeros 5 valores max por canal:", max_vals[:5].flatten())
        H_mag = H_mag/(max_vals + 1e-12) # [0,1]
        H_mag = 2 * H_mag - 1 # [-1,1]

        print("Rango final: min =", H_mag.min(), " max =", H_mag.max())
        all_channels.append(H_mag.astype(np.float32))
    # juntar todos los canales
    dataset = np.vstack(all_channels)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL DEL DATASET")
    print("=" * 60)
    print("Canales totales:", total_channels)
    print("Shape final:", dataset.shape)
    print("Min:", dataset.min(), " Max:", dataset.max())

    # guardar
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, dataset)

    print(f"\nDataset guardado en: {save_path}")
    return dataset

def analyze_dataset(data, save_dir="data", num_examples=5, heatmap_channels=100):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    N, L = data.shape
    print("\nDATASET ANALYSIS")
    print(f"Total de canales: {N}")
    print(f"Longitud temporal: {L}")
    print(f"Min: {data.min():.4f} | Max: {data.max():.4f}")

    # Ejemplos de canales reales:
    plt.figure(figsize=(10, 4))
    for i in range(num_examples):
        plt.plot(data[i], alpha=0.8)
    plt.title("Ejemplos de canales reales (magnitud normalizada)")
    plt.xlabel("Muestra temporal")
    plt.ylabel("Amplitud normalizada")
    plt.grid(alpha=0.3)

    if save_dir:
        plt.savefig(os.path.join(save_dir, "examples_channels.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Histograma global de amplitudes
    plt.figure(figsize=(6, 4))
    plt.hist(data.flatten(), bins=100, density=True, alpha=0.8)
    plt.title("Distribución global de amplitudes del canal")
    plt.xlabel("Amplitud normalizada")
    plt.ylabel("Densidad")
    plt.grid(alpha=0.3)

    if save_dir:
        plt.savefig(os.path.join(save_dir, "histogram_amplitudes.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Perfil temporal medio del canal
    mean_profile = data.mean(axis=0)
    std_profile = data.std(axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(mean_profile, label="Media")
    plt.fill_between(
        np.arange(L),
        mean_profile - std_profile,
        mean_profile + std_profile,
        alpha=0.3,
        label="±1σ"
    )
    plt.title("Perfil temporal medio del canal")
    plt.xlabel("Muestra temporal")
    plt.ylabel("Amplitud media")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_dir:
        plt.savefig(os.path.join(save_dir, "mean_channel_profile.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Heatmap de canales
    idx = np.random.choice(N, min(heatmap_channels, N), replace=False)
    subset = data[idx]

    plt.figure(figsize=(8, 6))
    plt.imshow(subset, aspect="auto", cmap="viridis")
    plt.colorbar(label="Amplitud normalizada")
    plt.title(f"Heatmap de canales reales ({subset.shape[0]} ejemplos)")
    plt.xlabel("Muestra temporal")
    plt.ylabel("Canal")

    if save_dir:
        plt.savefig(os.path.join(save_dir, "heatmap_channels.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Energía por canal
    energy = np.sum(data ** 2, axis=1)

    plt.figure(figsize=(6, 4))
    plt.hist(energy, bins=100, alpha=0.8)
    plt.title("Distribución de energía por canal")
    plt.xlabel("Energía normalizada")
    plt.ylabel("Número de canales")
    plt.grid(alpha=0.3)

    if save_dir:
        plt.savefig(os.path.join(save_dir, "channel_energy.png"), dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # load_nist_data("data/NIST_Samples.mat")
    #
    data = np.load("data/dataset_nist.npy")
    #
    # print("Shape:", data.shape)
    # print("Tipo:", data.dtype)
    # print("Primer canal:", data[0])

    analyze_dataset(data)