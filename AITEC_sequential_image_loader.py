import os
from PIL import Image
import torch
import numpy as np

class SequentialImageLoader:
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
    FUNCTION = "load_sequential_image"
    CATEGORY = "image"

    def load_sequential_image(self, folder_path, seed, include_subfolders):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Get image list
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

        # Sort files alphabetically
        image_files.sort()
        
        # Use seed as index, handle out of range
        index = seed % len(image_files)  # 範囲外でもループ
        
        # Select image based on index
        selected_image = image_files[index]

        # Load image
        image = Image.open(selected_image).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        # Generate mask
        mask = Image.open(selected_image).convert('L')
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)

        # Print to console
        print(f"Selected file: {os.path.relpath(selected_image, folder_path)} (Seed: {seed}, Index: {index})")

        # Increment seed for next execution
        return {
            "result": (image_tensor, mask_tensor),
            "hidden": {"seed": seed + 1}
        }

NODE_CLASS_MAPPINGS = {
    "SequentialImageLoader": SequentialImageLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SequentialImageLoader": "AITEC Sequential Image Loader"
}