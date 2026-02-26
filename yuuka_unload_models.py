"""
Yuuka Unload Models node.

Forces ComfyUI to fully unload all loaded models from both VRAM AND system RAM.
The key technique: moves model weights to 'meta' device (0 bytes) so that even
when ComfyUI's execution cache keeps references to the ModelPatcher, the actual
parameter/buffer tensors are replaced with zero-memory meta tensors.

Supports:
- Standard fp16/fp32 models (SD 1.5, SDXL, Flux, WAN2, etc.)
- Quantized models (GGUF, NF4, FP4, bitsandbytes 4/8-bit)
- All nn.Module-based models in ComfyUI's model cache

Designed to be placed between two KSampler nodes that use different models.
"""

import gc
import os
import ctypes
import torch
import logging

import comfy.model_management as mm

try:
    from comfy.comfy_types import IO
    ANY_TYPE = IO.ANY
except ImportError:
    ANY_TYPE = "*"

try:
    from server import PromptServer
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False


class YuukaUnloadModels:
    """
    Unloads all models from VRAM AND RAM between pipeline stages.

    The critical insight: ComfyUI's execution cache holds references to
    ModelPatcher objects, which hold the nn.Module containing all weights.
    Simply removing from current_loaded_models doesn't free those weights.

    Solution: Move the nn.Module to 'meta' device, which replaces all
    parameter tensors with zero-memory meta tensors while keeping the
    reference chain valid. Then GC + working set trim finishes the job.

    Handles quantized models (GGUF, NF4, bitsandbytes) with safe fallbacks.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "passthrough": (ANY_TYPE,),
            },
            "optional": {
                "enable": ("BOOLEAN", {"default": True}),
                "clear_cache": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "unload"
    CATEGORY = "Yuuka"
    DESCRIPTION = """
🧹 Fully unloads all models from VRAM AND RAM between pipeline stages.

Place between KSampler (High) output and KSampler (Low) input
to free the High model before loading the Low model.

Supports standard, quantized (GGUF, NF4, bitsandbytes), and custom models.

Options:
• enable: Toggle cleanup on/off
• clear_cache: Also reset execution cache (nodes will re-execute next run)
"""

    # ── Helper: detect quantized layers ──────────────────────────────
    @staticmethod
    def _has_quantized_layers(model):
        """Check if model contains quantized layers that can't use .to('meta')."""
        # Check for bitsandbytes quantized layers
        try:
            import bitsandbytes as bnb
            bnb_types = (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)
            for module in model.modules():
                if isinstance(module, bnb_types):
                    return True
        except (ImportError, Exception):
            pass

        # Check for GGUF/GGML custom tensors
        try:
            for p in model.parameters():
                ptype = type(p.data).__name__
                if ptype in ('GGMLTensor', 'GGUFParameter', 'GGUFWeight'):
                    return True
                if hasattr(p, 'gguf_quantization_type'):
                    return True
                # Break early after checking a few — no need to scan all
                break
        except Exception:
            pass

        # Check for common quantization module class names
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

    # ── Helper: estimate memory of a parameter ──────────────────────
    @staticmethod
    def _get_param_memory(p):
        """Get memory usage of a parameter, handling various tensor types."""
        try:
            if p.data.device.type == 'meta':
                return 0
        except Exception:
            return 0

        # Standard path — works for normal tensors
        try:
            return p.data.nbytes
        except Exception:
            pass

        # Fallback for quantized tensors
        try:
            return p.data.nelement() * p.data.element_size()
        except Exception:
            pass

        # Last resort: try untyped storage
        try:
            return p.data.untyped_storage().nbytes()
        except Exception:
            pass

        return 0

    # ── Helper: free a single parameter safely ──────────────────────
    @staticmethod
    def _free_param(p):
        """
        Free a single parameter, handling quantized types.
        Returns bytes freed.
        """
        try:
            if p.data.device.type == 'meta':
                return 0
        except Exception:
            return 0

        mem = YuukaUnloadModels._get_param_memory(p)

        # Clean up bitsandbytes quantization state attached to param
        for attr in ('quant_state', 'CB', 'SCB', 'absmax', 'code', 'blocksize'):
            if hasattr(p, attr):
                try:
                    delattr(p, attr)
                except Exception:
                    pass

        # Strategy 1: Replace with meta tensor (works for standard tensors)
        try:
            p.data = torch.empty(0, device='meta')
            return mem
        except Exception:
            pass

        # Strategy 2: Resize storage to 0 bytes (works for most tensor types)
        try:
            p.data.storage().resize_(0)
            return mem
        except Exception:
            pass

        # Strategy 3: Try with explicit dtype for exotic tensors
        try:
            p.data = torch.empty(0, dtype=torch.float16, device='meta')
            return mem
        except Exception:
            pass

        # Strategy 4: Untyped storage resize (last resort)
        try:
            p.data.untyped_storage().resize_(0)
            return mem
        except Exception:
            pass

        return 0

    # ── Helper: free a single buffer safely ─────────────────────────
    @staticmethod
    def _free_buffer(b):
        """Free a single buffer. Returns bytes freed."""
        try:
            if b.data.device.type == 'meta':
                return 0
        except Exception:
            return 0

        mem = YuukaUnloadModels._get_param_memory(b)

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

    # ── Main: free an entire model ──────────────────────────────────
    def _safe_free_model(self, model):
        """
        Free model weights using the safest strategy for the model type.
        Returns (freed_bytes, model_name, strategy_used).
        """
        model_name = model.__class__.__name__
        is_quantized = False

        try:
            is_quantized = self._has_quantized_layers(model)
        except Exception:
            # If detection itself fails, assume quantized (safer path)
            is_quantized = True

        # ── Strategy A: Direct .to('meta') for standard models ──
        # This is the fast path — one call moves everything at once.
        # Only used when we're confident there are no quantized layers.
        if not is_quantized:
            try:
                param_mem = sum(
                    self._get_param_memory(p) for p in model.parameters()
                )
                buf_mem = sum(
                    self._get_param_memory(b) for b in model.buffers()
                )
                model.to(device='meta')
                return (param_mem + buf_mem, model_name, "direct")
            except Exception:
                # Fall through to manual strategy
                pass

        # ── Strategy B: Parameter-by-parameter (quantized or failed direct) ──
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

    # ── Node execution ──────────────────────────────────────────────
    def unload(self, passthrough, enable=True, clear_cache=False):
        if not enable:
            return (passthrough,)

        import psutil
        process = psutil.Process(os.getpid())

        ram_before = process.memory_info().rss / (1024 ** 3)
        sys_ram_before = psutil.virtual_memory().percent
        vram_before = 0
        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated() / (1024 ** 3)

        print("=" * 60)
        print("[YuukaUnloadModels] Starting full memory cleanup...")
        print(f"  RAM before:  {ram_before:.2f} GB (system {sys_ram_before:.1f}%)")
        print(f"  VRAM before: {vram_before:.2f} GB")

        # ── Step 1: Collect actual nn.Module references BEFORE unloading ──
        # We need these to move weights to meta device after detaching
        real_models = []
        try:
            for loaded in mm.current_loaded_models:
                patcher = loaded.model  # ModelPatcher (via weakref)
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

        # ── Step 3: Move model weights to meta device (smart strategy) ──
        # THIS is the key step that actually frees RAM!
        # The execution cache still holds ModelPatcher → nn.Module references,
        # so GC alone can't free the weights. But moving to 'meta' device
        # replaces every parameter/buffer tensor with a zero-memory meta tensor.
        #
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

        # ── Step 4: Optional execution cache reset ──
        if clear_cache:
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

                # K32EmptyWorkingSet (modern Windows API)
                try:
                    kernel32.K32EmptyWorkingSet.argtypes = [wintypes.HANDLE]
                    kernel32.K32EmptyWorkingSet.restype = wintypes.BOOL
                    result = kernel32.K32EmptyWorkingSet(handle)
                    if result:
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

                # SetProcessWorkingSetSize with -1, -1 to trim
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
        gc.collect()  # one more pass
        ram_after = process.memory_info().rss / (1024 ** 3)
        sys_ram_after = psutil.virtual_memory().percent
        vram_after = 0
        if torch.cuda.is_available():
            vram_after = torch.cuda.memory_allocated() / (1024 ** 3)

        print(f"  RAM after:   {ram_after:.2f} GB (system {sys_ram_after:.1f}%)")
        print(f"  VRAM after:  {vram_after:.2f} GB")
        print(f"  RAM freed:   {ram_before - ram_after:.2f} GB")
        print(f"  VRAM freed:  {vram_before - vram_after:.2f} GB")
        print("[YuukaUnloadModels] Cleanup complete!")
        if clear_cache:
            print("  → Execution cache will be cleared after prompt finishes.")
            print("  → Next run: all nodes will re-execute (similar to restart).")
        print("=" * 60)

        return (passthrough,)


NODE_CLASS_MAPPINGS = {
    "YuukaUnloadModels": YuukaUnloadModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuukaUnloadModels": "🧹 Unload Models (Between Steps)",
}
