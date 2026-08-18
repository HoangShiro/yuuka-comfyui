import torch
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import requests

class Base64ToImage_Yuuka:
    """
    Yuuka's Custom Node:
    Nhận chuỗi Base64 đại diện cho hình ảnh trực tiếp từ API payload,
    giải mã trong bộ nhớ và chuyển đổi sang dạng IMAGE tensor cho ComfyUI.
    Node này KHÔNG lưu bất kỳ file nào xuống đĩa.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_base64": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "decode_base64"
    CATEGORY = "yuuka-comfyui/Tools"

    def decode_base64(self, image_base64):
        if not image_base64 or not image_base64.strip():
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            black_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            return (black_image, black_mask)

        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_np = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(img_np)[None,]

        if 'A' in img.getbands():
            mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask)
        else:
            mask_tensor = torch.zeros((img.height, img.width), dtype=torch.float32)
            
        mask_tensor = mask_tensor[None,]

        return (image_tensor, mask_tensor)


class ImageFromUrl_Yuuka:
    """
    Yuuka's Custom Node:
    Tải ảnh từ một URL hoặc API endpoint trực tiếp vào bộ nhớ
    và chuyển đổi thành IMAGE tensor cho ComfyUI.
    Node này KHÔNG lưu bất kỳ file nào xuống đĩa.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"multiline": False, "default": "http://"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_from_url"
    CATEGORY = "yuuka-comfyui/Tools"

    def load_from_url(self, image_url):
        if not image_url or image_url == "http://":
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            black_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            return (black_image, black_mask)

        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content))
        img = ImageOps.exif_transpose(img)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_np = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(img_np)[None,]
        
        if 'A' in img.getbands():
            mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask)
        else:
            mask_tensor = torch.zeros((img.height, img.width), dtype=torch.float32)
            
        mask_tensor = mask_tensor[None,]

        return (image_tensor, mask_tensor)


NODE_CLASS_MAPPINGS = {
    "Base64ToImage_Yuuka": Base64ToImage_Yuuka,
    "ImageFromUrl_Yuuka": ImageFromUrl_Yuuka
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Base64ToImage_Yuuka": "Yuuka Base64 to Image (API)",
    "ImageFromUrl_Yuuka": "Yuuka Image from URL (API)"
}
