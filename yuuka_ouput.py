# --- NEW FILE: ComfyUI/custom_nodes/yuuka_nodes/api_nodes.py ---
import torch
import numpy as np
from PIL import Image
import io
import base64

class ImageToBase64_Yuuka:
    """
    Yuuka's Custom Node:
    Chuyển đổi tensor hình ảnh thành chuỗi Base64 và trả về trong API output.
    Node này không lưu bất kỳ file nào xuống đĩa.
    """
    
    # Đánh dấu đây là một node output, kết quả của nó sẽ được đưa vào /history
    OUTPUT_NODE = True

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "encode_base64"
    CATEGORY = "Yuuka Tools"

    def encode_base64(self, images):
        results = []
        # images là một batch tensor, ta cần xử lý từng ảnh một
        for image in images:
            # Chuyển đổi tensor Pytorch về dạng numpy array
            # Định dạng: [H, W, C], giá trị 0-255, kiểu uint8
            i = 255. * image.cpu().numpy()
            img_np = np.clip(i, 0, 255).astype(np.uint8)
            
            # Tạo ảnh PIL từ numpy array
            pil_img = Image.fromarray(img_np)

            # Lưu ảnh vào một buffer trong bộ nhớ thay vì file
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG") # Có thể dùng WEBP hoặc JPEG nếu muốn
            img_bytes = buffer.getvalue()
            
            # Mã hoá bytes thành chuỗi base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            results.append(img_base64)

        # Trả về kết quả dưới dạng dictionary, ComfyUI sẽ tự động đưa nó vào API
        # Key "images_base64" là do chúng ta tự định nghĩa.
        return { "ui": { "images_base64": results } }

# Dictionary để ComfyUI đăng ký node khi khởi động
NODE_CLASS_MAPPINGS = {
    "ImageToBase64_Yuuka": ImageToBase64_Yuuka
}

# Tên hiển thị của node trong menu ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToBase64_Yuuka": "Yuuka Image to Base64 (API)"
}