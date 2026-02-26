"""
Yuuka custom nodes package entry point.

Runs the Git auto-update once and aggregates node mappings from submodules so
ComfyUI can discover them when the package is imported.
"""

from __future__ import annotations

from .yuuka_auto_update import ensure_auto_update


ensure_auto_update()

# Import node-providing modules so their classes are registered.
from . import yuuka_lora_downloader as _lora_downloader  # noqa: E402
from . import yuuka_ouput as _output_nodes  # noqa: E402
from . import yuuka_restart_backend as _restart_backend  # noqa: E402
from . import yuuka_unload_models as _unload_models  # noqa: E402
from . import yuuka_free_all_memory as _free_all_memory  # noqa: E402
from . import yuuka_video_output as _video_output  # noqa: E402


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _merge(module) -> None:
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))


_merge(_lora_downloader)
_merge(_output_nodes)
_merge(_restart_backend)
_merge(_unload_models)
_merge(_free_all_memory)
_merge(_video_output)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
