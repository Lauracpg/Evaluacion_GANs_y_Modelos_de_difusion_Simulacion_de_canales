import json
from train_modelos.train_DDPM import train as train_ddpm
from train_modelos.train_DDIM import train as train_ddim
from train_modelos.train_GAN_Conv1D import train as train_gan
from train_modelos.train_WGAN_Conv1D import train as train_wgan
from train_modelos.train_DCGAN_Conv1D import train as train_dcgan

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

def train(config_path):
    config = load_config(config_path)
    model_type = config["experiment"]["model_type"]

    if model_type == "ddpm":
        train_ddpm(config_path)

    elif model_type == "ddim":
        train_ddim(config_path)

    elif model_type == "gan":
        train_gan(config_path)

    elif model_type == "dcgan":
        train_dcgan(config_path)

    elif model_type == "wgan":
        train_wgan(config_path)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")