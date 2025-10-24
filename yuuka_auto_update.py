"""
Lightweight bootstrap to ensure the git auto-update runs at most once.

Each custom node module can import this file and invoke `ensure_auto_update()`
without worrying about module name collisions or duplicate git fetches.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_NAME = "yuuka_comfyui._internal_update"


def _load_update_module():
    """Load the update module anchored to this repository."""
    module = sys.modules.get(_MODULE_NAME)
    if module:
        return module

    update_path = Path(__file__).resolve().parent / "update.py"
    if not update_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(update_path))
    if not spec or not spec.loader:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001
        # Clean up the partially loaded module to avoid poisoning sys.modules.
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


def ensure_auto_update() -> None:
    """Trigger the repo auto-update logic, if available."""
    try:
        module = _load_update_module()
        if module and hasattr(module, "auto_update_if_needed"):
            module.auto_update_if_needed()
    except Exception as exc:  # noqa: BLE001
        print(f"[Yuuka Auto Update] Skipping due to unexpected error: {exc}")


__all__ = ["ensure_auto_update"]
