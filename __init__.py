"""
Project config helpers.

``POLICY_DIR``  – absolute path of the repo root.
``OUTPUTS_DIR`` – outputs/ subdirectory; all generated artefacts live here.
``CONFIGS_DIR`` – configs/ subdirectory; all JSON config files live here.
"""

from project_config import CONFIGS_DIR, OUTPUTS_DIR, POLICY_DIR, load_config

__all__ = ["POLICY_DIR", "OUTPUTS_DIR", "CONFIGS_DIR", "load_config"]
