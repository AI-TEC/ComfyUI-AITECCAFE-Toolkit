import os
import random
from PIL import Image
import torch
import numpy as np

class RandomImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "D:/ComfyUI/input", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_random_image"
    CATEGORY = "image"

    def load_random_image(self, folder_path, seed, include_subfolders):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # サブフォルダを含むか否かで画像リストを取得
        image_files = []
        if include_subfolders:
            for root, _, files in os.walk(folder_path):
                image_files.extend(os.path.join(root, f) for f in files 
                                 if os.path.splitext(f.lower())[1] in valid_extensions)
        else:
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                          if os.path.splitext(f.lower())[1] in valid_extensions]
        
        if not image_files:
            raise ValueError(f"No valid images found in folder: {folder_path}")

        random.seed(seed)
        selected_image = random.choice(image_files)

        # 画像を読み込み
        image = Image.open(selected_image).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        # マスクを生成（グレースケール）
        mask = Image.open(selected_image).convert('L')
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)

        # コンソールにファイル名とシードを表示
        print(f"Selected file: {os.path.relpath(selected_image, folder_path)} (Seed: {seed})")

        return (image_tensor, mask_tensor)

NODE_CLASS_MAPPINGS = {
    "RandomImageLoader": RandomImageLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomImageLoader": "AITEC Random Image Loader"
}