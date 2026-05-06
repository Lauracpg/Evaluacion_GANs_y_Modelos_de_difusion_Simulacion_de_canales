import numpy as np

def qpsk_mapper(bits):
    # Agrupar bits de 2 en 2 (QPSK usa 2 bits por símbolo)
    bits = bits.reshape(-1, 2)
    # Convertir bits a valores ±1
    # 0 → +1, 1 → -1
    symbols = (1 - 2*bits[:, 0]) + 1j * (1 - 2*bits[:, 1])
    # Normalizar energía del símbolo
    return symbols / np.sqrt(2)

def qpsk_demapper(symbols):
    # Si parte real es negativa → bit 1
    bits_real = (np.real(symbols) < 0).astype(int)
    # Si parte imaginaria es negativa → bit 1
    bits_imag = (np.imag(symbols) < 0).astype(int)
    # Recombinar en vector de bits plano
    return np.stack([bits_real, bits_imag], axis=1).reshape(-1)

def add_awgn(signal, snr_db):
    # Convertir SNR de dB a escala lineal
    snr = 10 ** (snr_db / 10)
    # Potencia media de la señal
    power = np.mean(np.abs(signal) ** 2)
    # Potencia del ruido según SNR
    noise_power = power / snr

    # Ruido complejo gaussiano (real + imaginario)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) +
        1j * np.random.randn(*signal.shape)
    )
    # Señal + ruido
    return signal + noise

def compute_ber(h, bits, snr_db=10):
    """
    Simula transmisión QPSK sobre un canal complejo
    y calcula tasa de error de bits (BER)
    """

    N, L = h.shape # número de señales y taps

    # 1. Mapear bits a símbolos QPSK
    x = qpsk_mapper(bits)

    # 2. Ajustar tamaño al canal
    num_symbols = N * L
    x = x[:num_symbols].reshape(N, L)

    # 3. Transformar canal a frecuencia
    H = np.fft.fft(h, axis=1)
    # 4. Transmisión por canal
    y = H * x

    # 5. Añadir ruido
    y = add_awgn(y, snr_db)

    # 6. Ecualización (invertir canal)
    x_hat = y / (H + 1e-12)

    # 7. Demodulación
    rx_bits = qpsk_demapper(x_hat.reshape(-1))

    # 8. Comparar bits transmitidos vs recibidos
    bits = bits[:len(rx_bits)]
    ber = np.mean(rx_bits != bits)

    return ber