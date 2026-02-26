# --- NEW FILE: ComfyUI/custom_nodes/yuuka-comfyui/yuuka_video_output.py ---
# Yuuka's Custom Node: Video to Base64
# Clone biến thể của VHS_VideoCombine, nhưng thay vì save video ra file,
# nó encode video frames thành webm và trả dữ liệu dưới dạng Base64
# để phù hợp truyền dữ liệu video tới "character-gallery" server.

import torch
import numpy as np
import subprocess
import io
import base64
import os
import sys
import shutil

class VideoToBase64_Yuuka:
    """
    Yuuka's Custom Node:
    Nhận batch IMAGE (video frames), encode thành video webm (VP9) bằng ffmpeg,
    rồi trả về chuỗi Base64 trong API output.
    Node này KHÔNG lưu bất kỳ file nào xuống đĩa.
    """
    
    OUTPUT_NODE = True

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 16, "min": 1, "max": 60, "step": 1}),
                "crf": ("INT", {"default": 20, "min": 0, "max": 63, "step": 1}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "encode_video_base64"
    CATEGORY = "yuuka-comfyui/Tools"

    @staticmethod
    def _find_ffmpeg():
        """Tìm đường dẫn ffmpeg, ưu tiên trong ComfyUI dir."""
        # 1. Thử tìm trong thư mục ComfyUI (Windows thường dùng cách này)
        comfyui_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        possible_paths = [
            os.path.join(comfyui_dir, "ffmpeg.exe"),
            os.path.join(comfyui_dir, "bin", "ffmpeg.exe"),
            os.path.join(comfyui_dir, "ffmpeg", "ffmpeg.exe"),
        ]
        for p in possible_paths:
            if os.path.isfile(p):
                return p
        
        # 2. Thử PATH hệ thống
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path
        
        # 3. Fallback
        return "ffmpeg"

    def encode_video_base64(self, images, frame_rate=16, crf=20, audio=None):
        """
        Encode batch images thành video webm VP9 rồi trả về base64.
        """
        if images is None or (isinstance(images, torch.Tensor) and images.size(0) == 0):
            return {"ui": {"video_base64": [], "format": ["video/webm"]}}

        ffmpeg_path = self._find_ffmpeg()
        fps = max(1, int(frame_rate))
        
        # Chuẩn bị frames - chuyển tensor sang numpy uint8
        if isinstance(images, torch.Tensor):
            frames_np = (images.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            frames_np = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    arr = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = np.array(img)
                frames_np.append(arr)
            frames_np = np.array(frames_np)
        
        num_frames, height, width, channels = frames_np.shape
        
        # Đảm bảo width và height là số chẵn (yêu cầu của VP9)
        if width % 2 != 0:
            width = width - 1
            frames_np = frames_np[:, :, :width, :]
        if height % 2 != 0:
            height = height - 1
            frames_np = frames_np[:, :height, :, :]

        # Chạy ffmpeg để encode thành webm VP9
        cmd = [
            ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',  # stdin
            '-c:v', 'libvpx-vp9',
            '-crf', str(crf),
            '-b:v', '0',
            '-pix_fmt', 'yuv420p',
            '-f', 'webm',
            '-'  # stdout
        ]

        try:
            # Chuyển frames_np thành raw bytes
            raw_bytes = frames_np.tobytes()
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
            )
            
            video_bytes, stderr_output = process.communicate(input=raw_bytes, timeout=300)
            
            if process.returncode != 0:
                error_msg = stderr_output.decode('utf-8', errors='ignore')[-500:]
                print(f"[VideoToBase64_Yuuka] FFmpeg error (code {process.returncode}): {error_msg}")
                return {"ui": {"video_base64": [], "format": "video/webm", "error": f"FFmpeg failed: {error_msg[:200]}"}}
            
            if not video_bytes:
                print("[VideoToBase64_Yuuka] FFmpeg produced no output")
                return {"ui": {"video_base64": [], "format": "video/webm", "error": "FFmpeg produced no output"}}
            
            # Encode video bytes thành base64
            video_b64 = base64.b64encode(video_bytes).decode('utf-8')
            
            return {
                "ui": {
                    "video_base64": [video_b64],
                    "format": ["video/webm"],
                    "frame_count": [num_frames],
                    "frame_rate": [fps],
                    "width": [width],
                    "height": [height],
                }
            }
        except subprocess.TimeoutExpired:
            process.kill()
            print("[VideoToBase64_Yuuka] FFmpeg timed out after 300s")
            return {"ui": {"video_base64": [], "format": ["video/webm"], "error": ["FFmpeg timed out"]}}
        except FileNotFoundError:
            print(f"[VideoToBase64_Yuuka] FFmpeg not found at: {ffmpeg_path}")
            return {"ui": {"video_base64": [], "format": ["video/webm"], "error": [f"FFmpeg not found: {ffmpeg_path}"]}}
        except Exception as e:
            print(f"[VideoToBase64_Yuuka] Error: {e}")
            return {"ui": {"video_base64": [], "format": ["video/webm"], "error": [str(e)]}}


NODE_CLASS_MAPPINGS = {
    "VideoToBase64_Yuuka": VideoToBase64_Yuuka
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoToBase64_Yuuka": "Yuuka Video to Base64 (API)"
}
