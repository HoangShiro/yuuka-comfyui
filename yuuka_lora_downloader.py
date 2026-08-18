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

    # Hard safety cap: auto-cancel a task that runs longer than this (seconds)
    DEFAULT_MAX_SECONDS = 180

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
    CATEGORY = "yuuka-comfyui/Loaders"
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

        # Establish a hard deadline for the entire task (fail-safe against hanging jobs)
        deadline = time.monotonic() + self.DEFAULT_MAX_SECONDS

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

        loras_dir = _get_primary_loras_directory()
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
                deadline,
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
        except TimeoutError as exc:
            # Specific handling for deadline exceed: remove partial and report cancelled
            error_msg = f"Task cancelled after timeout: {exc}"
            print(f"[Yuuka Lora Downloader] {error_msg}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            self._emit_status(
                tracking_id,
                "cancelled",
                "Download cancelled due to 3-minute timeout.",
                filename=lora_filename,
            )
            return ("Timeout: download cancelled",)
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
        deadline: float,
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
                    # Enforce wall-clock deadline regardless of network activity
                    if time.monotonic() > deadline:
                        self._emit_status(
                            tracking_id,
                            "cancelled",
                            "Cancelling: exceeded 3-minute limit.",
                            filename=filename,
                            bytes_downloaded=downloaded,
                            total_bytes=expected_bytes,
                        )
                        raise TimeoutError("exceeded 180 seconds")
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

    def _cache_thumbnail(self, preview_url: str, lora_dir: str, stem: str, headers: dict = None):
        """Download and cache the LoRA preview thumbnail into its dedicated folder."""
        if not preview_url:
            return None
        try:
            os.makedirs(lora_dir, exist_ok=True)
            thumb_path = os.path.join(lora_dir, f"{stem}.jpg")
            thumb_generic = os.path.join(lora_dir, "thumbnail.jpg")

            if os.path.isfile(thumb_path) and os.path.getsize(thumb_path) > 0:
                return thumb_path

            resp = requests.get(preview_url, headers=headers or {}, timeout=25)
            if resp.status_code == 200 and len(resp.content) > 0:
                with open(thumb_path, "wb") as handle:
                    handle.write(resp.content)
                with open(thumb_generic, "wb") as handle:
                    handle.write(resp.content)
                print(f"[Yuuka Lora Downloader] Cached thumbnail for '{stem}' in {lora_dir}.")
                return thumb_path
        except Exception as exc:
            print(f"[Yuuka Lora Downloader] Failed to cache thumbnail for '{stem}': {exc}")
        return None

    def _save_metadata(self, model_data: dict, loras_dir: str, lora_filename: str, headers: dict = None):
        stem = os.path.splitext(lora_filename)[0]
        # Dedicated folder for each LoRA's metadata and thumbnail assets
        lora_dir = os.path.join(loras_dir, stem)
        os.makedirs(lora_dir, exist_ok=True)

        # 1. Save metadata into LoRA's dedicated folder
        dedicated_info_path = os.path.join(lora_dir, f"{stem}.json")
        dedicated_meta_path = os.path.join(lora_dir, "metadata.json")
        legacy_info_path = os.path.join(loras_dir, f"{stem}.json")

        try:
            with open(dedicated_info_path, "w", encoding="utf-8") as handle:
                json.dump(model_data, handle, indent=4)
            with open(dedicated_meta_path, "w", encoding="utf-8") as handle:
                json.dump(model_data, handle, indent=4)
            # Also maintain legacy root sidecar for backwards compatibility
            with open(legacy_info_path, "w", encoding="utf-8") as handle:
                json.dump(model_data, handle, indent=4)
            print(f"[Yuuka Lora Downloader] Metadata saved in '{lora_dir}'.")
        except Exception as exc:
            print(f"[Yuuka Lora Downloader] Failed to write metadata: {exc}")

        # 2. Extract and cache thumbnail image into the dedicated folder
        preview_url = ""
        versions = model_data.get("modelVersions", [])
        if versions and isinstance(versions, list) and isinstance(versions[0], dict):
            imgs = versions[0].get("images", [])
            if imgs and isinstance(imgs, list) and isinstance(imgs[0], dict):
                preview_url = imgs[0].get("url") or ""
        if not preview_url and model_data.get("preview_url"):
            preview_url = model_data.get("preview_url")

        if preview_url:
            self._cache_thumbnail(preview_url, lora_dir, stem, headers=headers)


NODE_CLASS_MAPPINGS = {
    "Yuuka_Lora_Downloader": YuukaLoraDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuuka_Lora_Downloader": "Lora Downloader (by Yuuka)",
}


# ----------------------------
# Additional server endpoints
# ----------------------------

def _get_loras_directories() -> list[str]:
    """Return all configured LoRA directories, ensuring user documents folder is included."""
    paths = folder_paths.get_folder_paths("loras") or []
    user_doc_path = os.path.normpath("C:/Users/nyaac/Documents/ComfyUI/models/loras")
    normalized = []
    for p in paths:
        if isinstance(p, str) and p.strip():
            np = os.path.normpath(p)
            if np not in normalized and os.path.isdir(np):
                normalized.append(np)
    if os.path.isdir(user_doc_path) and user_doc_path not in normalized:
        normalized.append(user_doc_path)
    return normalized if normalized else [user_doc_path]


def _get_primary_loras_directory() -> str:
    """Return the preferred directory for saving new LoRAs."""
    dirs = _get_loras_directories()
    for d in dirs:
        if "documents" in d.lower() and "comfyui" in d.lower():
            return d
    return dirs[-1] if dirs else os.path.normpath("C:/Users/nyaac/Documents/ComfyUI/models/loras")


async def _delete_lora_files(filename: str) -> dict:
    """Delete a LoRA .safetensors file, its dedicated folder, and sidecar metadata across all loras dirs."""
    result = {"deleted": False, "filename": filename, "removed": [], "errors": []}
    if not filename:
        result["errors"].append("Missing filename")
        return result

    base_name = os.path.basename(filename).strip()
    stem = os.path.splitext(base_name)[0]

    all_dirs = _get_loras_directories()
    deleted_any_model = False

    for loras_dir in all_dirs:
        # Possible candidate paths for the model file
        candidate_safetensors = [
            os.path.join(loras_dir, base_name),
            os.path.join(loras_dir, f"{stem}.safetensors"),
            os.path.join(loras_dir, f"{base_name}.safetensors"),
            os.path.join(loras_dir, stem, f"{stem}.safetensors"),
            os.path.join(loras_dir, stem, base_name),
        ]

        for target_path in candidate_safetensors:
            try:
                if os.path.isfile(target_path):
                    os.remove(target_path)
                    result["removed"].append(target_path)
                    deleted_any_model = True
            except Exception as exc:
                result["errors"].append(f"Failed to delete {target_path}: {exc}")

        # Delete root sidecars if exist
        sidecar_candidates = [
            os.path.join(loras_dir, f"{stem}.json"),
            os.path.join(loras_dir, f"{stem}.metadata.json"),
            os.path.join(loras_dir, f"{base_name}.json"),
            os.path.join(loras_dir, f"{base_name}.metadata.json"),
        ]
        for p in sidecar_candidates:
            try:
                if os.path.isfile(p):
                    os.remove(p)
                    result["removed"].append(p)
            except Exception as exc:
                result["errors"].append(f"Failed to delete sidecar JSON: {exc}")

        # Delete dedicated folder and its files if exists
        lora_dir = os.path.join(loras_dir, stem)
        try:
            if os.path.isdir(lora_dir):
                import shutil
                shutil.rmtree(lora_dir, ignore_errors=True)
                result["removed"].append(lora_dir)
        except Exception as exc:
            result["errors"].append(f"Failed to remove LoRA directory {lora_dir}: {exc}")

    result["deleted"] = deleted_any_model or len(result["removed"]) > 0
    return result


@PromptServer.instance.routes.post("/yuuka/lora/delete")
async def yuuka_lora_delete(request):
    """HTTP endpoint to delete a LoRA by filename on the ComfyUI host."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    filename = (payload.get("filename") or "").strip()
    if not filename:
        return web.json_response({"deleted": False, "error": "Missing filename"}, status=400)

    result = await _delete_lora_files(filename)
    status = 200 if result.get("deleted") or "LoRA file not found" in result.get("errors", []) else 500
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

    all_dirs = _get_loras_directories()
    status_map = {}

    for entry in filenames:
        if not isinstance(entry, str):
            continue
        cleaned = entry.strip()
        if not cleaned:
            continue
        safe_name = os.path.basename(cleaned)
        stem = os.path.splitext(safe_name)[0]

        is_present = False
        for loras_dir in all_dirs:
            candidates = [
                os.path.join(loras_dir, safe_name),
                os.path.join(loras_dir, f"{stem}.safetensors"),
                os.path.join(loras_dir, stem, f"{stem}.safetensors"),
            ]
            if any(os.path.isfile(c) for c in candidates):
                is_present = True
                break
        status_map[safe_name] = is_present

    return web.json_response({"status": status_map}, status=200)


@PromptServer.instance.routes.get("/yuuka/lora/list")
async def yuuka_lora_list(_request):
    """Return all LoRA filenames available on disk across all directories."""
    all_dirs = _get_loras_directories()
    all_files = set()

    for loras_dir in all_dirs:
        try:
            for name in os.listdir(loras_dir):
                if isinstance(name, str) and name.lower().endswith(".safetensors"):
                    all_files.add(name)
        except Exception:
            pass

    sorted_files = sorted(all_files)
    return web.json_response({"files": sorted_files}, status=200)


@PromptServer.instance.routes.get("/yuuka/lora/thumbnail")
async def yuuka_lora_thumbnail(request):
    """Serve local cached thumbnail image for a LoRA, downloading on-demand if needed."""
    filename = request.query.get("filename", "").strip()
    if not filename:
        return web.Response(status=400, text="Missing filename")

    safe_name = os.path.basename(filename)
    stem = os.path.splitext(safe_name)[0]
    all_dirs = _get_loras_directories()

    # 1. Search candidate paths in all directories
    for loras_dir in all_dirs:
        lora_dir = os.path.join(loras_dir, stem)
        candidates = [
            os.path.join(lora_dir, f"{stem}.jpg"),
            os.path.join(lora_dir, "thumbnail.jpg"),
            os.path.join(lora_dir, f"{stem}.png"),
            os.path.join(lora_dir, "thumbnail.png"),
            os.path.join(lora_dir, "preview.png"),
            os.path.join(loras_dir, f"{stem}.preview.png"),
            os.path.join(loras_dir, f"{stem}.png"),
            os.path.join(loras_dir, f"{stem}.jpg"),
        ]
        for path in candidates:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                content_type = "image/png" if path.lower().endswith(".png") else "image/jpeg"
                return web.FileResponse(path, headers={"Content-Type": content_type, "Cache-Control": "max-age=86400"})

    # 2. On-demand caching if metadata exists
    for loras_dir in all_dirs:
        lora_dir = os.path.join(loras_dir, stem)
        meta_candidates = [
            os.path.join(lora_dir, f"{stem}.json"),
            os.path.join(lora_dir, "metadata.json"),
            os.path.join(loras_dir, f"{stem}.json"),
            os.path.join(loras_dir, f"{stem}.metadata.json"),
        ]
        for meta_path in meta_candidates:
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    preview_url = ""
                    versions = data.get("modelVersions", [])
                    if versions and isinstance(versions, list) and isinstance(versions[0], dict):
                        imgs = versions[0].get("images", [])
                        if imgs and isinstance(imgs, list) and isinstance(imgs[0], dict):
                            preview_url = imgs[0].get("url") or ""
                    if not preview_url and data.get("preview_url"):
                        preview_url = data.get("preview_url")

                    if preview_url:
                        os.makedirs(lora_dir, exist_ok=True)
                        thumb_path = os.path.join(lora_dir, f"{stem}.jpg")
                        resp = requests.get(preview_url, timeout=20)
                        if resp.status_code == 200 and len(resp.content) > 0:
                            with open(thumb_path, "wb") as f:
                                f.write(resp.content)
                            with open(os.path.join(lora_dir, "thumbnail.jpg"), "wb") as f:
                                f.write(resp.content)
                            return web.FileResponse(thumb_path, headers={"Content-Type": "image/jpeg", "Cache-Control": "max-age=86400"})
                except Exception as exc:
                    print(f"[Yuuka Lora Downloader] On-demand thumbnail download error for '{stem}': {exc}")

    return web.Response(status=404, text="Thumbnail not found")


@PromptServer.instance.routes.get("/yuuka/lora/models")
async def yuuka_lora_models(_request):
    """Return all LoRAs available on disk with their sidecar metadata and cached thumbnail URLs."""
    all_dirs = _get_loras_directories()
    models = {}

    for loras_dir in all_dirs:
        try:
            for filename in os.listdir(loras_dir):
                if not isinstance(filename, str) or not filename.lower().endswith(".safetensors"):
                    continue
                if filename in models:
                    continue

                stem = os.path.splitext(filename)[0]
                lora_dir = os.path.join(loras_dir, stem)

                # Metadata candidate paths
                meta_candidates = [
                    os.path.join(lora_dir, f"{stem}.json"),
                    os.path.join(lora_dir, "metadata.json"),
                    os.path.join(loras_dir, f"{stem}.json"),
                    os.path.join(loras_dir, f"{stem}.metadata.json"),
                ]

                model_record = {
                    "filename": filename,
                    "name": stem,
                    "civitai_url": "",
                    "thumbnail": "",
                    "remote_thumbnail": "",
                    "base_model": "",
                    "trained_words": [],
                    "folder": stem,
                    "available": True,
                }

                data = None
                found_meta_path = None
                for p in meta_candidates:
                    if os.path.isfile(p):
                        try:
                            with open(p, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                found_meta_path = p
                                break
                        except Exception:
                            pass

                if isinstance(data, dict):
                    if found_meta_path and not os.path.isfile(os.path.join(lora_dir, "metadata.json")):
                        try:
                            os.makedirs(lora_dir, exist_ok=True)
                            with open(os.path.join(lora_dir, "metadata.json"), "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=4)
                        except Exception:
                            pass

                    model_id = data.get("id")
                    if model_id:
                        model_record["civitai_url"] = f"https://civitai.com/models/{model_id}"
                    elif data.get("civitai_url"):
                        model_record["civitai_url"] = data.get("civitai_url")

                    if data.get("name"):
                        model_record["name"] = data.get("name")

                    versions = data.get("modelVersions", [])
                    if versions and isinstance(versions, list) and isinstance(versions[0], dict):
                        v0 = versions[0]
                        model_record["base_model"] = v0.get("baseModel") or v0.get("base_model") or ""
                        words = v0.get("trainedWords", [])
                        if isinstance(words, list):
                            model_record["trained_words"] = words
                        elif isinstance(words, str):
                            model_record["trained_words"] = [w.strip() for w in words.split(",") if w.strip()]
                        imgs = v0.get("images", [])
                        if imgs and isinstance(imgs, list) and isinstance(imgs[0], dict):
                            model_record["remote_thumbnail"] = imgs[0].get("url") or ""

                    if not model_record["remote_thumbnail"] and data.get("preview_url"):
                        model_record["remote_thumbnail"] = data.get("preview_url")
                    if not model_record["base_model"] and data.get("base_model") and data.get("base_model") != "Unknown":
                        model_record["base_model"] = data.get("base_model")
                    if not model_record["trained_words"] and data.get("trained_words"):
                        model_record["trained_words"] = data.get("trained_words")

                    # Local thumbnail endpoint
                    model_record["thumbnail"] = f"/yuuka/lora/thumbnail?filename={filename}"

                models[filename] = model_record
        except Exception:
            pass

    return web.json_response({"models": models}, status=200)

