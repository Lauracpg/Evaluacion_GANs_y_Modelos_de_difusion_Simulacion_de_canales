# Evaluación de GANs y Modelos de Difusión para la Simulación de Canales

## Descripción

Este proyecto estudia la generación sintética de canales de comunicaciones inalámbricas mediante modelos generativos basados en aprendizaje profundo. Se implementan y comparan distintas arquitecturas:

* GAN
* DCGAN
* WGAN-GP
* DDPM
* DDIM

El objetivo es evaluar la capacidad de estos modelos para reproducir las características estadísticas y físicas de canales reales mediante métricas como:

* Power Delay Profile (PDP)
* Densidad espectral de potencia (PSD)
* Autocorrelación
* Retardo medio
* RMS Delay Spread
* Bit Error Rate (BER)

---
## Estructura del proyecto

```text
project
├── preprocess_dataset.py
├── train_router.py
├── gan_run_sweep.py
├── dm_run_sweep.py
│
├── train_modelos
│   ├── train_GAN_Conv1D.py
│   ├── train_DCGAN_Conv1D.py
│   ├── train_WGAN_Conv1D.py
│   ├── train_DDPM.py
│   └── train_DDIM.py
│
├── evaluation_modelos
│   ├── evaluation.py
│   └── BER.py
│
├── config
    ├── gans_config.json
    ├── dm_config.json
    ├── gans_sweep_config.json
    ├── dm_sweep_config.json
    └── evaluation_config.json
```
---
## Librerías utilizadas

* PyTorch
* NumPy
* SciPy
* Matplotlib
* json
* argparse
* importlib
* sys
* copy
* itertools
* os

---
## Flujo de ejecución

### 1. Preprocesamiento

```bash
python preprocess_dataset.py
```

Genera un dataset normalizado en formato:

```text
(N, 2, L)
```

donde:

* `N`: número de muestras
* `L`: longitud temporal del canal
* `2`: componentes I/Q

---

### 2. Entrenamiento individual

Para entrenar una arquitectura concreta:

```bash
python train_GAN/DCGAN/WGAN_Conv1D.py config/gans_config.json
```

o

```bash
python train_DDPM/DDIM.py config/dm_config.json
```

Los modelos entrenados se almacenan automáticamente en el directorio configurado.


---

### 3. Barridos de hiperparámetros

Para GAN, DCGAN y WGAN-GP:

```bash
python gan_run_sweep.py
```

Para DDPM y DDIM:

```bash
python dm_run_sweep.py
```

Los scripts generan automáticamente todas las combinaciones de hiperparámetros especificadas en los archivos de configuración.

---

### 4. Evaluación de modelos

```bash
python evaluation_modelos/evaluation.py \
    --model path/to/best_model.pth \
    --model_type gan
```

Durante la evaluación se generan nuevas realizaciones sintéticas del canal y se comparan con muestras reales.

Las métricas calculadas incluyen:

* PDP
* PSD
* Autocorrelación
* Retardo medio
* RMS Delay Spread
* BER

---

## Resultados

Para cada experimento se guardan:

* Checkpoints del modelo
* Configuración utilizada
* Métricas de evaluación
* Gráficas generadas
* Resultados BER