import json
import copy
import itertools
import os
from train_router import train

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

def flatten_grid(grid):
    keys = []
    values = []

    for section, params in grid.items():
        for k, v in params.items():
            keys.append((section, k))
            values.append(v)

    return keys, values

def apply_config(base, keys, combo):
    cfg = copy.deepcopy(base)

    for (section, key), value in zip(keys, combo):
        cfg[section][key] = value

    return cfg

def build_experiment_name(cfg):
    return (
        f"{cfg['experiment']['model_type']}"
        f"_T{cfg['diffusion']['T']}"
        f"_lr{cfg['training']['lr']}"
        f"_te{cfg['model']['time_emb_dim']}"
    )

if __name__ == "__main__":
    base = load_json("config/dm_config.json")
    grid = load_json("config/dm_sweep_config.json")

    keys, values = flatten_grid(grid)

    combinations = list(itertools.product(*values))
    print(f"Total experiments: {len(combinations)}")

    for combo in combinations:
        cfg = apply_config(base, keys, combo)

        exp_name = build_experiment_name(cfg)

        save_dir = f"checkpoints/datos_loopback/sweep_ddim_2/{exp_name}"
        os.makedirs(save_dir, exist_ok=True)

        done_flag = os.path.join(save_dir, "done.txt")
        if os.path.exists(done_flag):
            print(f"Skipping {exp_name}")
            continue

        cfg["paths"]["save_dir"] = save_dir
        cfg["experiment"]["name"] = exp_name

        config_path = os.path.join(save_dir, "config.json")
        save_json(cfg, config_path)

        print(f"\nRunning {exp_name}")

        try:
            train(config_path)

            with open(done_flag, "w") as f:
                f.write("done")

        except Exception as e:
            print(f"Error en {exp_name}: {e}")