"""
Yuuka Free All Memory node (OUTPUT_NODE).

Aggressively frees ALL memory (VRAM + RAM) after the workflow completes.
Achieves near-restart levels of memory cleanup without actually restarting.

Supports:
- Standard fp16/fp32 models (SD 1.5, SDXL, Flux, WAN2, etc.)
- Quantized models (GGUF, NF4, FP4, bitsandbytes 4/8-bit)
- All nn.Module-based models in ComfyUI's model cache

Flexible placement:
- As a START node: no input needed, output connects to downstream nodes
- As an END node: connect any output to 'trigger', runs at the end
- STANDALONE: just place it and run, no connections needed
"""

import gc
import os
import ctypes
import torch
import logging

import comfy.model_management as mm

try:
    from server import PromptServer
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False

try:
    from comfy.comfy_types import IO
    ANY_TYPE = IO.ANY
except ImportError:
    ANY_TYPE = "*"


class YuukaFreeAllMemory:
    """
    Aggressively frees ALL memory after the workflow completes.

    This is an OUTPUT_NODE that always executes when connected.
    Place at the very end of your workflow to clean up everything.

    Achieves near-restart memory state by:
    1. Moving all model weights to meta device (frees RAM)
    2. Clearing ComfyUI's model tracking list
    3. Signaling execution cache reset (processed after prompt finishes)
    4. Emptying GPU cache (CUDA + MPS) and running garbage collection
    5. Trimming Windows process working set

    Handles quantized models (GGUF, NF4, bitsandbytes) with safe fallbacks.
    """

    # ── Reuse smart model-freeing helpers from YuukaUnloadModels ─────
    @staticmethod
    def _has_quantized_layers(model):
        """Check if model contains quantized layers that can't use .to('meta')."""
        try:
            import bitsandbytes as bnb
            bnb_types = (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)
            for module in model.modules():
                if isinstance(module, bnb_types):
                    return True
        except (ImportError, Exception):
            pass

        try:
            for p in model.parameters():
                ptype = type(p.data).__name__
                if ptype in ('GGMLTensor', 'GGUFParameter', 'GGUFWeight'):
                    return True
                if hasattr(p, 'gguf_quantization_type'):
                    return True
                break
        except Exception:
            pass

        try:
            for module in model.modules():
                mtype = type(module).__name__.lower()
                if any(q in mtype for q in (
                    'quantized', 'gguf', 'ggml', 'nf4', 'fp4',
                    'int8linear', 'int4linear', 'fp8linear',
                )):
                    return True
        except Exception:
            pass

        return False

    @staticmethod
    def _get_param_memory(p):
        """Get memory usage of a parameter, handling various tensor types."""
        try:
            if p.data.device.type == 'meta':
                return 0
        except Exception:
            return 0
        try:
            return p.data.nbytes
        except Exception:
            pass
        try:
            return p.data.nelement() * p.data.element_size()
        except Exception:
            pass
        try:
            return p.data.untyped_storage().nbytes()
        except Exception:
            pass
        return 0

    @staticmethod
    def _free_param(p):
        """Free a single parameter, handling quantized types."""
        try:
            if p.data.device.type == 'meta':
                return 0
        except Exception:
            return 0

        mem = YuukaFreeAllMemory._get_param_memory(p)

        for attr in ('quant_state', 'CB', 'SCB', 'absmax', 'code', 'blocksize'):
            if hasattr(p, attr):
                try:
                    delattr(p, attr)
                except Exception:
                    pass

        for strategy in [
            lambda: setattr(p, 'data', torch.empty(0, device='meta')),
            lambda: p.data.storage().resize_(0),
            lambda: setattr(p, 'data', torch.empty(0, dtype=torch.float16, device='meta')),
            lambda: p.data.untyped_storage().resize_(0),
        ]:
            try:
                strategy()
                return mem
            except Exception:
                continue
        return 0

    @staticmethod
    def _free_buffer(b):
        """Free a single buffer."""
        try:
            if b.data.device.type == 'meta':
                return 0
        except Exception:
            return 0
        mem = YuukaFreeAllMemory._get_param_memory(b)
        try:
            b.data = torch.empty(0, device='meta')
            return mem
        except Exception:
            pass
        try:
            b.data.storage().resize_(0)
            return mem
        except Exception:
            pass
        return 0

    def _safe_free_model(self, model):
        """Free model weights using the safest strategy for the model type."""
        model_name = model.__class__.__name__
        is_quantized = False
        try:
            is_quantized = self._has_quantized_layers(model)
        except Exception:
            is_quantized = True

        if not is_quantized:
            try:
                param_mem = sum(self._get_param_memory(p) for p in model.parameters())
                buf_mem = sum(self._get_param_memory(b) for b in model.buffers())
                model.to(device='meta')
                return (param_mem + buf_mem, model_name, "direct")
            except Exception:
                pass

        freed = 0
        failed_count = 0
        for name, p in model.named_parameters():
            try:
                f = self._free_param(p)
                freed += f
                if f == 0 and self._get_param_memory(p) > 0:
                    failed_count += 1
            except Exception:
                failed_count += 1
        for name, b in model.named_buffers():
            try:
                f = self._free_buffer(b)
                freed += f
                if f == 0 and self._get_param_memory(b) > 0:
                    failed_count += 1
            except Exception:
                failed_count += 1

        strategy = "quantized-aware" if is_quantized else "manual-fallback"
        if failed_count > 0:
            strategy += f" ({failed_count} skipped)"
        return (freed, model_name, strategy)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "trigger": (ANY_TYPE,),
                "enable": ("BOOLEAN", {"default": True}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "free_all"
    CATEGORY = "Yuuka"
    DESCRIPTION = """
🔥 Aggressively frees ALL memory (VRAM + RAM).
Achieves near-restart memory state without restarting the backend.

Flexible placement — works in 3 modes:
• START: Place at the beginning, connect output to downstream nodes
• END: Connect any output to 'trigger' input
• STANDALONE: No connections needed, just run it

After cleanup, ComfyUI will be in a near-fresh state:
- All model weights freed from RAM
- VRAM fully cleared
- Execution cache reset (nodes will re-execute on next run)
- Windows working set trimmed
"""

    def free_all(self, trigger=None, enable=True):
        if not enable:
            out = trigger if trigger is not None else True
            return {"ui": {"text": ["Cleanup disabled"]}, "result": (out,)}

        import psutil
        process = psutil.Process(os.getpid())

        ram_before = process.memory_info().rss / (1024 ** 3)
        sys_ram_before = psutil.virtual_memory().percent
        vram_before = 0
        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated() / (1024 ** 3)

        print("=" * 60)
        print("[YuukaFreeAllMemory] Starting full memory cleanup...")
        print(f"  RAM before:  {ram_before:.2f} GB (system {sys_ram_before:.1f}%)")
        print(f"  VRAM before: {vram_before:.2f} GB")

        # ── Step 1: Collect all nn.Module references ──
        real_models = []
        try:
            for loaded in mm.current_loaded_models:
                patcher = loaded.model
                if patcher is not None and hasattr(patcher, 'model') and patcher.model is not None:
                    real_models.append(patcher.model)
        except Exception as e:
            print(f"  [!] Error collecting model refs: {e}")

        # ── Step 2: Unload from current_loaded_models ──
        try:
            count = len(mm.current_loaded_models)
            for i in range(count - 1, -1, -1):
                try:
                    mm.current_loaded_models[i].model_unload()
                except Exception:
                    pass

            while len(mm.current_loaded_models) > 0:
                x = mm.current_loaded_models.pop()
                del x

            print(f"  [✓] Cleared {count} models from ComfyUI model cache")
        except Exception as e:
            print(f"  [!] Model cache cleanup error: {e}")

        # ── Step 3: Move ALL model weights to meta device (smart strategy) ──
        # For quantized models (GGUF, bitsandbytes, NF4), .to('meta') may fail,
        # so we fall back to parameter-by-parameter freeing with multiple strategies.
        total_freed_mb = 0
        for model in real_models:
            try:
                freed_bytes, model_name, strategy = self._safe_free_model(model)
                freed_mb = freed_bytes / (1024 ** 2)
                total_freed_mb += freed_mb
                if freed_mb > 0:
                    print(f"  [✓] Freed {model_name} via {strategy} ({freed_mb:.0f} MB)")
                else:
                    print(f"  [~] {model_name}: no freeable weights found")
            except Exception as e:
                print(f"  [!] Failed to free {model.__class__.__name__}: {e}")
        del real_models

        if total_freed_mb > 0:
            print(f"  [✓] Total model data freed: {total_freed_mb:.0f} MB")

        # ── Step 4: Signal execution cache reset ──
        # As an OUTPUT_NODE at the end of the workflow, this flag is processed
        # right after prompt execution completes → clears all cached node outputs
        # (including model loader outputs), achieving near-restart state
        try:
            if HAS_SERVER and hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                server = PromptServer.instance
                if hasattr(server, 'prompt_queue'):
                    server.prompt_queue.set_flag("unload_models", True)
                    server.prompt_queue.set_flag("free_memory", True)
                    print("  [✓] Signaled execution cache reset (runs after prompt)")
        except Exception as e:
            print(f"  [!] Cache reset signal failed: {e}")

        # ── Step 5: Soft empty ComfyUI cache ──
        try:
            mm.soft_empty_cache()
            print("  [✓] Soft emptied ComfyUI cache")
        except Exception:
            pass

        # ── Step 6: Clear GPU cache (CUDA + MPS) ──
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("  [✓] Emptied CUDA cache")
        elif hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                print("  [✓] Emptied MPS cache")
            except Exception:
                pass

        # ── Step 7: Aggressive garbage collection ──
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                print(f"  [✓] GC pass {i+1}: collected {collected} objects")

        # ── Step 8: Final GPU cleanup ──
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Step 9: Windows - trim process working set ──
        if os.name == 'nt':
            try:
                from ctypes import wintypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetCurrentProcess()

                try:
                    kernel32.K32EmptyWorkingSet.argtypes = [wintypes.HANDLE]
                    kernel32.K32EmptyWorkingSet.restype = wintypes.BOOL
                    kernel32.K32EmptyWorkingSet(handle)
                    print("  [✓] Windows: Emptied working set")
                except Exception:
                    try:
                        psapi = ctypes.windll.psapi
                        psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
                        psapi.EmptyWorkingSet.restype = wintypes.BOOL
                        psapi.EmptyWorkingSet(handle)
                        print("  [✓] Windows: Emptied working set (psapi)")
                    except Exception:
                        pass

                try:
                    kernel32.SetProcessWorkingSetSize.argtypes = [
                        wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t
                    ]
                    kernel32.SetProcessWorkingSetSize.restype = wintypes.BOOL
                    kernel32.SetProcessWorkingSetSize(
                        handle,
                        ctypes.c_size_t(-1),
                        ctypes.c_size_t(-1)
                    )
                    print("  [✓] Windows: Trimmed working set size")
                except Exception as e:
                    print(f"  [!] Windows trim: {e}")
            except Exception as e:
                print(f"  [!] Windows cleanup error: {e}")

        # ── Final stats ──
        gc.collect()
        ram_after = process.memory_info().rss / (1024 ** 3)
        sys_ram_after = psutil.virtual_memory().percent
        vram_after = 0
        if torch.cuda.is_available():
            vram_after = torch.cuda.memory_allocated() / (1024 ** 3)

        ram_freed = ram_before - ram_after
        vram_freed = vram_before - vram_after

        print(f"  RAM after:   {ram_after:.2f} GB (system {sys_ram_after:.1f}%)")
        print(f"  VRAM after:  {vram_after:.2f} GB")
        print(f"  RAM freed:   {ram_freed:.2f} GB")
        print(f"  VRAM freed:  {vram_freed:.2f} GB")
        if total_freed_mb > 0:
            print(f"  Model data:  {total_freed_mb:.0f} MB moved to meta")
        print("[YuukaFreeAllMemory] Cleanup complete!")
        print("  → Execution cache will be cleared after prompt finishes.")
        print("  → Next run: all nodes will re-execute (similar to restart).")
        print("=" * 60)

        ui_text = (
            f"RAM: {ram_before:.1f} → {ram_after:.1f} GB (freed {ram_freed:.1f} GB) | "
            f"VRAM: {vram_before:.1f} → {vram_after:.1f} GB (freed {vram_freed:.1f} GB) | "
            f"System: {sys_ram_before:.0f}% → {sys_ram_after:.0f}%"
        )

        # Pass through trigger if provided, otherwise output True as signal
        out = trigger if trigger is not None else True
        return {"ui": {"text": [ui_text]}, "result": (out,)}


NODE_CLASS_MAPPINGS = {
    "YuukaFreeAllMemory": YuukaFreeAllMemory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuukaFreeAllMemory": "🔥 Free All Memory (Like Restart)",
}
