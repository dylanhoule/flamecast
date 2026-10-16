import yaml

REQUIRED = {"grid_size", "cell_size_m", "stencil", "max_iters", "arrival_cap_h", "fuels", "curing"}


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = REQUIRED - cfg.keys()
    if missing:
        raise ValueError(f"{path} missing keys: {sorted(missing)}")
    if cfg["stencil"] not in (8, 16):
        raise ValueError(f"stencil must be 8 or 16, got {cfg['stencil']}")
    return cfg
