import copy

import pytest

from dgforecast.config import load_config


@pytest.fixture
def small_cfg():
    """A tiny version of the real config so tests run in seconds."""
    cfg = copy.deepcopy(load_config())
    cfg["data"].update(n_stores=4, n_items=6, n_days=500, future_days=28, seed=7)
    cfg["validation"]["min_rows"] = 1000
    cfg["train"]["num_boost_round"] = 200
    return cfg
