import os
from PIL import Image
import torch
import numpy as np
import cv2

class SequentialMediaLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "D:/ComfyUI/input", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "load_all_frames": ("BOOLEAN", {"default": False}),
                "max_frames": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1}),
                "frame_step": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("image", "mask", "filename", "total_frames")
    FUNCTION = "load_sequential_media"
    CATEGORY = "image"

    def load_sequential_media(self, folder_path, seed, include_subfolders, frame_index, load_all_frames, max_frames, frame_step):
        # 有効な拡張子（画像と動画）
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        valid_extensions = image_extensions + video_extensions
        
        # メディアファイルリストを取得
        media_files = []
        if include_subfolders:
            for root, _, files in os.walk(folder_path):
                media_files.extend(os.path.join(root, f) for f in files 
                                if os.path.splitext(f.lower())[1] in valid_extensions)
        else:
            media_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                          if os.path.splitext(f.lower())[1] in valid_extensions]
        
        if not media_files:
            raise ValueError(f"No valid media files found in folder: {folder_path}")

        # ファイルを名前順にソート
        media_files.sort()
        
        # seedをインデックスとして使用
        index = seed % len(media_files)
        selected_file = media_files[index]
        
        # ファイル拡張子を確認
        file_ext = os.path.splitext(selected_file.lower())[1]
        filename = os.path.relpath(selected_file, folder_path)
        
        if file_ext in image_extensions:
            # 画像ファイルの処理
            image, mask = self._load_image(selected_file)
            total_frames = 1
            print(f"Selected image: {filename} (Seed: {seed}, Index: {index})")
            
        elif file_ext in video_extensions:
            if load_all_frames:
                # 動画の全フレームを読み込み
                image, mask, total_frames = self._load_all_video_frames(selected_file, max_frames, frame_step)
                print(f"Selected video (all frames): {filename} (Seed: {seed}, Index: {index}, Loaded frames: {image.shape[0]}/{total_frames})")
            else:
                # 単一フレームを読み込み
                image, mask, total_frames = self._load_video_frame(selected_file, frame_index)
                print(f"Selected video (single frame): {filename} (Seed: {seed}, Index: {index}, Frame: {frame_index}/{total_frames-1})")
        
        return (image, mask, filename, total_frames)
    
    def _load_image(self, image_path):
        """画像ファイルをロード"""
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        # マスクを生成（グレースケール変換）
        mask = Image.open(image_path).convert('L')
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)[None,]
        
        return image_tensor, mask_tensor
    
    def _load_video_frame(self, video_path, frame_index):
        """動画ファイルから指定フレームをロード"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # 総フレーム数を取得
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        # フレームインデックスを範囲内に調整
        frame_index = frame_index % total_frames
        
        # 指定フレームに移動
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_index} from video: {video_path}")
        
        # BGR → RGB変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # numpy配列をテンソルに変換
        image_np = frame_rgb.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        # マスクを生成（グレースケール変換）
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_np = frame_gray.astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)[None,]
        
        return image_tensor, mask_tensor, total_frames
    
    def _load_all_video_frames(self, video_path, max_frames, frame_step):
        """動画ファイルからすべてのフレームをロード"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # 総フレーム数を取得
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        frames = []
        masks = []
        frame_count = 0
        current_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # frame_stepに従ってフレームをスキップ
            if current_frame % frame_step == 0:
                # BGR → RGB変換
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_np = frame_rgb.astype(np.float32) / 255.0
                frames.append(frame_np)
                
                # マスクを生成（グレースケール変換）
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask_np = frame_gray.astype(np.float32) / 255.0
                masks.append(mask_np)
                
                frame_count += 1
                
                # 最大フレーム数に達したら終了
                if frame_count >= max_frames:
                    break
            
            current_frame += 1
        
        cap.release()
        
        if not frames:
            raise ValueError(f"Could not load any frames from video: {video_path}")
        
        # リストをテンソルに変換
        image_tensor = torch.from_numpy(np.array(frames))
        mask_tensor = torch.from_numpy(np.array(masks))
        
        print(f"Loaded {len(frames)} frames from video (step: {frame_step})")
        
        return image_tensor, mask_tensor, total_frames

NODE_CLASS_MAPPINGS = {
    "SequentialMediaLoader": SequentialMediaLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SequentialMediaLoader": "AITEC Sequential Media Loader"
}