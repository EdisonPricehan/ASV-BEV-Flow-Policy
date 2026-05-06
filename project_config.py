"""
Project config helpers.

``POLICY_DIR``  – absolute path of the repo root.
``OUTPUTS_DIR`` – outputs/ subdirectory; all generated artefacts live here.
``CONFIGS_DIR`` – configs/ subdirectory; all JSON config files live here.
"""
from pathlib import Path
import json

POLICY_DIR: Path = Path(__file__).parent
OUTPUTS_DIR: Path = POLICY_DIR / "outputs"
CONFIGS_DIR: Path = POLICY_DIR / "configs"


def load_config(*names: str) -> dict:
    """
    Load and merge one or more JSON configs from ``configs/``.

    Parameters
    ----------
    *names : str
        Config file stems to load in order, e.g. ``"common"``, ``"train"``.
        Each file is merged (shallow at the top level) on top of the previous.

    Returns
    -------
    dict  – merged configuration
    """
    merged: dict = {}
    for name in names:
        path = CONFIGS_DIR / f"{name}.json"
        with open(path) as f:
            data = json.load(f)
        # drop _comment* keys (documentation-only)
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        # drop _comment keys inside nested dicts too
        for key, val in data.items():
            if isinstance(val, dict):
                data[key] = {k: v for k, v in val.items() if not k.startswith("_")}
        merged.update(data)
    return merged

