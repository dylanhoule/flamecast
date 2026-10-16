import pytest

from sim.config import load_config


def test_default_config_loads():
    assert load_config("configs/default.yaml")["grid_size"] == 256


def test_missing_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("grid_size: 64\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_config(p)
