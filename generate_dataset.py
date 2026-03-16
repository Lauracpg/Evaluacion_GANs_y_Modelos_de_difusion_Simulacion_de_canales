import numpy as np
import os
import h5py
from matplotlib import pyplot as plt

def load_synthetic_data(file_path, L=128, save_path="data/datos_sinteticos/TDL_D_85ns_fd50000_SNR20_h_real.npy"):
    print(f"Cargando archivo: {file_path}")
    H = np.load(file_path) # (128, 100000)
    print("Shape original:", H.shape)
    print("dtype:", H.dtype)
    print("Primeros valores:\n", H[:5, :5])
    print("Min:", H.min(), "Max:", H.max())

    H= H[:L, :]
    H = H.T # (100000, 128)
    print(f"Shape canales: {H.shape}")

    H_real = np.real(H)
    H_imag = np.imag(H)

    # norm energética
    # energy = np.sqrt(np.sum(H_real**2 + H_imag**2, axis=1, keepdims=True))
    # H_real = H_real / (energy + 1e-12)
    # H_imag = H_imag / (energy + 1e-12)

    dataset = np.stack([H_real, H_imag], axis=1).astype(np.float32)
    print("Shape dataset:", dataset.shape)

    # [-1, 1] global
    max_abs = np.max(np.abs(dataset))
    dataset = dataset / (max_abs + 1e-12)

    print("Rango final:", dataset.min(), dataset.max())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, dataset)
    print("Dataset guardado en:", save_path)

    return dataset

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
        print(f"\nProcesando {ck} ...")

        H_raw = f[ck] # matriz 241 x YYY
        # conservar el dtype estructurado
        H_struct = H_raw[...]

        print("dtype:", H_struct.dtype)

        # reconstruir números complejos
        H_real = np.nan_to_num(H_struct["real"], nan=0.0)
        H_imag = np.nan_to_num(H_struct["imag"], nan=0.0)
        H = H_real + 1j * H_imag

        print("Shape original:", H.shape)

        # H es 241 x num_canales, a veces se invierte dependiendo de h5py
        if H.shape[0] != 241 and H.shape[1] == 241:
            print("La matriz está transpuesta. Corrigiendo orientación...")
            H = H.T

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

        # Separar real e imaginario (SIN normalizar todavía)
        H_real = np.real(H_crop)
        H_imag = np.imag(H_crop)

        # Normalización
        energy = np.sqrt(np.sum(H_real**2 + H_imag**2, axis=1, keepdims=True))
        H_real = H_real / (energy + 1e-12)
        H_imag = H_imag / (energy + 1e-12)


        H_2_channel = np.stack([H_real, H_imag], axis=1)
        all_channels.append(H_2_channel.astype(np.float32))

        print(f"Añadidos {num_c} canales")

    # juntar todos los canales
    dataset = np.vstack(all_channels)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL DEL DATASET")
    print("=" * 60)
    print("Canales totales:", total_channels)
    print("Shape final:", dataset.shape)

    # [-1, 1] normalización
    max_abs = np.max(np.abs(dataset))
    dataset = dataset / (max_abs + 1e-12)

    print("Max absoluto global:", max_abs)
    print("Rango final:", dataset.min(), dataset.max())

    # guardar
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, dataset)

    print(f"\nDataset guardado en: {save_path}")
    return dataset

def analyze_dataset(data, save_dir="data", num_examples=5, heatmap_channels=100):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    N, C, L = data.shape # C = 2
    assert C==2, "El dataset debe tener 2 canales (real, imag)"

    print("\nDATASET ANALYSIS")
    print(f"Total de canales: {N}")
    print(f"Canales por muestra: {C}")
    print(f"Longitud temporal: {L}")
    print(f"Min global: {data.min():.4f} | Max global: {data.max():.4f}")

    real = data[:, 0, :]
    imag = data[:, 1, :]

    # Ejemplos de canales reales:
    plt.figure(figsize=(10, 4))
    for i in range(min(num_examples, N)):
        plt.plot(real[i], alpha=0.7)
    plt.title("Ejemplos - Parte REAL")
    plt.xlabel("Muestra temporal")
    plt.ylabel("Amplitud")
    plt.grid(alpha=0.3)

    if save_dir:
        plt.savefig(os.path.join(save_dir, "examples_real.png"), dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 4))
    for i in range(min(num_examples, N)):
        plt.plot(imag[i], alpha=0.7)
    plt.title("Ejemplos - Parte IMAGINARIA")
    plt.xlabel("Muestra temporal")
    plt.ylabel("Amplitud")
    plt.grid(alpha=0.3)

    if save_dir:
        plt.savefig(os.path.join(save_dir, "examples_imag.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Histograma global de amplitudes
    plt.figure(figsize=(6, 4))
    plt.hist(real.flatten(), bins=100, density=True, alpha=0.6, label="Real")
    plt.hist(imag.flatten(), bins=100, density=True, alpha=0.6, label="Imag")
    plt.title("Distribución global Real / Imag")
    plt.xlabel("Valor normalizado")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_dir:
        plt.savefig(os.path.join(save_dir, "histogram_real_imag.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Magnitud
    magnitud = np.sqrt(real**2 + imag**2)
    plt.figure(figsize=(6, 4))
    plt.hist(magnitud.flatten(), bins=100, density=True, alpha=0.8)
    plt.title("Distribución de la magnitud |H|")
    plt.xlabel("Magnitud")
    plt.ylabel("Densidad")
    plt.grid(alpha=0.3)

    if save_dir:
        plt.savefig(os.path.join(save_dir, "histogram_magnitud.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Perfil temporal medio del canal
    mean_real = real.mean(axis=0)
    std_real = real.std(axis=0)

    mean_imag = imag.mean(axis=0)
    std_imag = imag.std(axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(mean_real, label="Media Real")
    plt.fill_between(
        np.arange(L),
        mean_real - std_real,
        mean_real + std_real,
        alpha=0.3,
    )
    plt.title("Perfil temporal medio - Parte Real")
    plt.xlabel("Muestra temporal")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_dir:
        plt.savefig(os.path.join(save_dir, "mean_real_profile.png"), dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(mean_imag, label="Media Imag")
    plt.fill_between(
        np.arange(L),
        mean_imag - std_imag,
        mean_imag + std_imag,
        alpha=0.3
    )
    plt.title("Perfil temporal medio - Parte Imaginaria")
    plt.xlabel("Muestra temporal")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_dir:
        plt.savefig(os.path.join(save_dir, "mean_imag_profile.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Heatmap (magnitud)
    idx = np.random.choice(N, min(heatmap_channels, N), replace=False)
    subset = magnitud[idx]

    plt.figure(figsize=(8, 6))
    plt.imshow(subset, aspect="auto", cmap="viridis")
    plt.colorbar(label="Magnitud")
    plt.title(f"Heatmap de magnitud ({subset.shape[0]} canales)")
    plt.xlabel("Muestra temporal")
    plt.ylabel("Canal")

    if save_dir:
        plt.savefig(os.path.join(save_dir, "heatmap_magnitud.png"), dpi=300, bbox_inches="tight")
    plt.show()

    print("\nAnálisis completado.")

if __name__ == "__main__":
    #data = load_nist_data("data/NIST_Samples.mat", L=128, save_path="data/dataset_nist.npy")
    #print("Shape:", data.shape)

    data = load_synthetic_data(
        "data/datos_sinteticos/TDL_D_85ns_fd1000_SNR20_h_real.npy",
        L=128,
        save_path="data/datos_sinteticos/dataset_synthetic.npy"
    )

    analyze_dataset(data)