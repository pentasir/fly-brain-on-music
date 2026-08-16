"""Real neuron positions (from CAVE's cell_representative_point table,
cached in data/connectome/neuron_positions.csv) -- the actual anatomical
coordinates used to place neurons in the 3D scene (see web_scene.py)."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "connectome"

_positions_cache = None


def load_positions() -> pd.DataFrame:
    global _positions_cache
    if _positions_cache is None:
        df = pd.read_csv(DATA_DIR / "neuron_positions.csv")
        _positions_cache = df.set_index("root_id")
    return _positions_cache


def _to_scene_coords(x_nm, y_nm, z_nm):
    """nm -> um, flip x for a consistent left/right reading."""
    return -x_nm / 1000.0, y_nm / 1000.0, z_nm / 1000.0
