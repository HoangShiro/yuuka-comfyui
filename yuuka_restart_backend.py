"""
Yuuka Restart Backend node.

Restarts the ComfyUI backend after the current task completes,
ensuring a clean RAM & VRAM state for subsequent tasks.
Replicates the same restart logic used by ComfyUI Manager's reboot.
"""

import os
import sys
import time
import threading

try:
    from comfy.comfy_types import IO
    ANY_TYPE = IO.ANY
except ImportError:
    ANY_TYPE = "*"


def _do_restart(delay: float):
    """
    Performs the actual restart after a delay.
    Runs in a separate thread so the node can return first,
    allowing ComfyUI to finish the current prompt cleanly.
    """
    time.sleep(delay)

    print("\n[YuukaRestartBackend] Restarting now...\n")

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    # Build the command to re-launch ComfyUI (same logic as ComfyUI Manager)
    sys_argv = sys.argv.copy()
    if "--windows-standalone-build" in sys_argv:
        sys_argv.remove("--windows-standalone-build")

    if sys_argv[0].endswith("__main__.py"):
        module_name = os.path.basename(os.path.dirname(sys_argv[0]))
        cmds = [sys.executable, "-m", module_name] + sys_argv[1:]
    elif sys.platform.startswith("win32"):
        cmds = ['"' + sys.executable + '"', '"' + sys_argv[0] + '"'] + sys_argv[1:]
    else:
        cmds = [sys.executable] + sys_argv

    print(f"[YuukaRestartBackend] Command: {cmds}", flush=True)

    # Replace the current process with a fresh ComfyUI instance
    os.execv(sys.executable, cmds)


class YuukaRestartBackend:
    """
    Restarts the ComfyUI backend after the current task completes.

    This is an OUTPUT_NODE so it WILL always execute when connected.
    Simply connect any output from your workflow to the 'trigger' input.

    After restart, ComfyUI will start fresh with clean RAM & VRAM.
    The frontend (browser) will automatically reconnect.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trigger": (ANY_TYPE,),
                "enable": ("BOOLEAN", {"default": True}),
                "delay_seconds": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Seconds to wait before restarting. "
                               "Gives ComfyUI time to finish saving outputs."
                }),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "restart_backend"
    CATEGORY = "Yuuka"
    DESCRIPTION = """
🔄 Restarts the ComfyUI backend after the current task completes.

This completely restarts ComfyUI, which:
- Frees ALL RAM and VRAM (guaranteed clean state)
- Reloads all custom nodes
- The browser will auto-reconnect after restart

Usage:
- Connect 'trigger' to ANY output from your last node
- enable: Toggle restart on/off without removing the node
- delay_seconds: Wait time before restart (default 3s)
"""

    def restart_backend(self, trigger=None, enable=True, delay_seconds=3.0):
        if not enable:
            print("[YuukaRestartBackend] Restart is DISABLED. Skipping.")
            return {"ui": {"text": ["Restart disabled"]}}

        print("=" * 60)
        print(f"[YuukaRestartBackend] ComfyUI will restart in {delay_seconds}s...")
        print("[YuukaRestartBackend] All RAM & VRAM will be freed on restart.")
        print("=" * 60)

        # Launch restart in a background thread so the node can return first.
        # This allows ComfyUI to finish saving outputs before the process restarts.
        t = threading.Thread(target=_do_restart, args=(delay_seconds,), daemon=True)
        t.start()

        return {"ui": {"text": [f"Restarting in {delay_seconds}s..."]}}


NODE_CLASS_MAPPINGS = {
    "YuukaRestartBackend": YuukaRestartBackend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuukaRestartBackend": "🔄 Restart Backend (After Task)",
}
