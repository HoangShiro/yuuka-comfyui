import os
import json
import re
import time
from typing import Optional

import requests
from dotenv import load_dotenv
import folder_paths
from server import PromptServer
from aiohttp import web


class YuukaLoraDownloader:
    """
    Custom node that downloads a LoRA from Civitai and keeps interested
    frontends informed about the progress by emitting websocket events.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "civitai_url": (
                    "STRING",
                    {"multiline": False, "default": "https://civitai.com/models/..."},
                )
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "tracking_id": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_name",)
    FUNCTION = "download_lora"
    CATEGORY = "yuuka_nodes/Loaders"
    OUTPUT_NODE = True

    def _get_api_key(self, api_key_input: str) -> str:
        if api_key_input:
            return api_key_input

        custom_nodes_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(custom_nodes_dir, ".env"),
            os.path.join(os.path.dirname(os.path.abspath(folder_paths.__file__)), ".env"),
        ]
        for path in candidates:
            if os.path.exists(path):
                load_dotenv(dotenv_path=path)
                found_key = os.getenv("CIVITAI_API_KEY")
                if found_key:
                    return found_key
        return ""

    def _emit_status(self, tracking_id: str, status: str, message: str, **extra):
        """Send progress/status updates to ComfyUI websocket clients."""
        if not tracking_id:
            return
        payload = {"tracking_id": tracking_id, "status": status, "message": message}
        payload.update(extra)
        try:
            PromptServer.instance.send_sync("yuuka.lora_downloader", payload)
        except Exception as exc:
            print(f"[Yuuka Lora Downloader] Failed to emit status: {exc}")

    def download_lora(self, civitai_url: str, api_key: str = "", tracking_id: str = ""):
        api_key = self._get_api_key(api_key)
        print(f"[Yuuka Lora Downloader] Begin request for URL: {civitai_url}")

        if not api_key:
            error_msg = "Loi: Khong tim thay CIVITAI_API_KEY"
            print(f"[Yuuka Lora Downloader] {error_msg}")
            self._emit_status(tracking_id, "error", error_msg)
            return (error_msg,)

        match = re.search(r"/models/(\d+)", civitai_url.strip())
        if not match:
            error_msg = "Loi: Link Civitai khong hop le."
            print(f"[Yuuka Lora Downloader] {error_msg} - {civitai_url}")
            self._emit_status(tracking_id, "error", error_msg)
            return (error_msg,)

        model_id = match.group(1)
        headers = {"Authorization": f"Bearer {api_key}"}
        details_url = f"https://civitai.com/api/v1/models/{model_id}"
        self._emit_status(
            tracking_id,
            "info",
            "Fetching model details",
            step="fetch_details",
            model_id=model_id,
        )

        try:
            model_resp = requests.get(details_url, headers=headers, timeout=15)
            model_resp.raise_for_status()
            model_data = model_resp.json()
        except requests.exceptions.RequestException as exc:
            error_msg = f"Loi: Khong lay duoc thong tin model: {exc}"
            print(f"[Yuuka Lora Downloader] {error_msg}")
            self._emit_status(tracking_id, "error", error_msg)
            return (error_msg,)

        model_type = (model_data.get("type") or "").lower()
        if model_type != "lora":
            error_msg = f"Loi: Model nay khong phai LORA (Loai: {model_type})."
            print(f"[Yuuka Lora Downloader] {error_msg}")
            self._emit_status(tracking_id, "error", error_msg, model_type=model_type)
            return (error_msg,)

        valid_file_info = self._select_file(model_data)
        if not valid_file_info:
            error_msg = "Loi: Khong tim thay file LoRA .safetensors nao de tai."
            print(f"[Yuuka Lora Downloader] {error_msg}")
            self._emit_status(tracking_id, "error", error_msg)
            return (error_msg,)

        lora_filename = valid_file_info["name"]
        download_url = valid_file_info["downloadUrl"]
        expected_size_kb = valid_file_info.get("sizeKB", 0)
        expected_bytes = int(expected_size_kb * 1024)

        loras_dir = folder_paths.get_folder_paths("loras")[0]
        file_path = os.path.join(loras_dir, lora_filename)

        self._emit_status(
            tracking_id,
            "info",
            "Preparing download",
            step="prepare_download",
            filename=lora_filename,
            expected_size_kb=expected_size_kb,
        )

        if os.path.exists(file_path) and self._is_same_size(file_path, expected_bytes):
            print(f"[Yuuka Lora Downloader] File '{lora_filename}' already up-to-date.")
            self._save_metadata(model_data, loras_dir, lora_filename)
            self._emit_status(
                tracking_id,
                "completed",
                "LoRA already available. Metadata refreshed.",
                filename=lora_filename,
                was_cached=True,
                model_data=model_data,
            )
            return (lora_filename,)

        try:
            print(f"[Yuuka Lora Downloader] Downloading '{lora_filename}' ...")
            self._perform_download(
                download_url,
                headers,
                file_path,
                tracking_id,
                lora_filename,
                expected_bytes,
            )
            print(f"[Yuuka Lora Downloader] Download finished for '{lora_filename}'.")
            self._save_metadata(model_data, loras_dir, lora_filename)
            self._emit_status(
                tracking_id,
                "completed",
                "Download completed.",
                filename=lora_filename,
                was_cached=False,
                model_data=model_data,
            )
            return (lora_filename,)
        except Exception as exc:
            error_msg = f"Loi khi tai/ghi file: {exc}"
            print(f"[Yuuka Lora Downloader] {error_msg}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            self._emit_status(tracking_id, "error", error_msg, filename=lora_filename)
            return (error_msg,)

    def _select_file(self, model_data: dict) -> Optional[dict]:
        for version in model_data.get("modelVersions", []):
            for file_obj in version.get("files", []):
                is_safetensors = file_obj.get("name", "").lower().endswith(".safetensors")
                if is_safetensors and file_obj.get("type") == "Model":
                    return file_obj
        return None

    def _is_same_size(self, file_path: str, expected_bytes: int) -> bool:
        if expected_bytes <= 0:
            return False
        actual_size = os.path.getsize(file_path)
        return abs(actual_size - expected_bytes) < 2048

    def _perform_download(
        self,
        download_url: str,
        headers: dict,
        file_path: str,
        tracking_id: str,
        filename: str,
        expected_bytes: int,
    ):
        downloaded = 0
        last_emit = time.monotonic()
        last_percent = -1

        self._emit_status(
            tracking_id,
            "downloading",
            "Download started.",
            filename=filename,
            total_bytes=expected_bytes,
        )

        with requests.get(download_url, headers=headers, stream=True, timeout=600) as response:
            response.raise_for_status()
            with open(file_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    percent = None
                    if expected_bytes > 0:
                        percent = int(min(downloaded / expected_bytes, 1.0) * 100)
                    now = time.monotonic()
                    if percent is None or percent != last_percent or (now - last_emit) > 0.75:
                        self._emit_status(
                            tracking_id,
                            "downloading",
                            "Downloading...",
                            filename=filename,
                            bytes_downloaded=downloaded,
                            total_bytes=expected_bytes,
                            progress_percent=percent,
                        )
                        last_emit = now
                        last_percent = percent if percent is not None else last_percent

    def _save_metadata(self, model_data: dict, loras_dir: str, lora_filename: str):
        info_path = os.path.join(loras_dir, f"{os.path.splitext(lora_filename)[0]}.json")
        try:
            with open(info_path, "w", encoding="utf-8") as handle:
                json.dump(model_data, handle, indent=4)
            print(f"[Yuuka Lora Downloader] Metadata saved for {lora_filename}.")
        except Exception as exc:
            print(f"[Yuuka Lora Downloader] Failed to write metadata: {exc}")


NODE_CLASS_MAPPINGS = {
    "Yuuka_Lora_Downloader": YuukaLoraDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuuka_Lora_Downloader": "Lora Downloader (by Yuuka)",
}


# ----------------------------
# Additional server endpoints
# ----------------------------

def _get_loras_directory() -> str:
    paths = folder_paths.get_folder_paths("loras")
    if not paths:
        raise RuntimeError("LoRA directory not configured")
    return paths[0]


async def _delete_lora_files(filename: str) -> dict:
    """Delete a LoRA .safetensors file and its sidecar .json metadata.

    Returns a dict with {deleted: bool, filename: str, removed: [paths], errors: [str]}
    """
    result = {"deleted": False, "filename": filename, "removed": [], "errors": []}
    if not filename:
        result["errors"].append("Missing filename")
        return result

    # Only allow basename to prevent path traversal
    safe_name = os.path.basename(filename)
    try:
        loras_dir = _get_loras_directory()
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"Cannot resolve loras path: {exc}")
        return result

    target_path = os.path.join(loras_dir, safe_name)
    # Sidecar metadata JSON placed next to LoRA file using stem
    sidecar_path = os.path.join(loras_dir, f"{os.path.splitext(safe_name)[0]}.json")

    # Delete LoRA file if exists
    try:
        if os.path.isfile(target_path):
            os.remove(target_path)
            result["removed"].append(target_path)
        else:
            result["errors"].append("LoRA file not found")
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"Failed to delete LoRA file: {exc}")

    # Delete sidecar JSON if exists (best-effort)
    try:
        if os.path.isfile(sidecar_path):
            os.remove(sidecar_path)
            result["removed"].append(sidecar_path)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"Failed to delete sidecar JSON: {exc}")

    result["deleted"] = any(p.endswith(safe_name) or p.endswith(f"{os.path.splitext(safe_name)[0]}.json") for p in result["removed"]) and ("LoRA file not found" not in result["errors"]) 
    return result


@PromptServer.instance.routes.post("/yuuka/lora/delete")
async def yuuka_lora_delete(request):
    """HTTP endpoint to delete a LoRA by filename on the ComfyUI host.

    Body JSON: { filename: str }
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    filename = (payload.get("filename") or "").strip()
    if not filename:
        return web.json_response({"deleted": False, "error": "Missing filename"}, status=400)

    result = await _delete_lora_files(filename)
    status = 200 if result.get("deleted") or "LoRA file not found" in result.get("errors", []) else 500
    # If the main file was not found, we still return 200 to indicate idempotent success on removal intent.
    if "LoRA file not found" in result.get("errors", []):
        status = 200
    return web.json_response(result, status=status)


@PromptServer.instance.routes.post("/yuuka/lora/status")
async def yuuka_lora_status(request):
    """Return availability status for a list of LoRA filenames."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    filenames = payload.get("filenames")
    if not isinstance(filenames, list):
        return web.json_response({"status": {}}, status=200)

    try:
        loras_dir = _get_loras_directory()
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"status": {}, "error": str(exc)}, status=500)

    status_map = {}
    for entry in filenames:
        if not isinstance(entry, str):
            continue
        cleaned = entry.strip()
        if not cleaned:
            continue
        safe_name = os.path.basename(cleaned)
        target_path = os.path.join(loras_dir, safe_name)
        status_map[safe_name] = os.path.isfile(target_path)

    return web.json_response({"status": status_map}, status=200)


@PromptServer.instance.routes.get("/yuuka/lora/list")
async def yuuka_lora_list(_request):
    """Return all LoRA filenames available on disk."""
    try:
        loras_dir = _get_loras_directory()
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"files": [], "error": str(exc)}, status=500)

    try:
        entries = [
            name
            for name in os.listdir(loras_dir)
            if isinstance(name, str) and name.lower().endswith(".safetensors")
        ]
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"files": [], "error": str(exc)}, status=500)

    entries.sort()
    return web.json_response({"files": entries}, status=200)
