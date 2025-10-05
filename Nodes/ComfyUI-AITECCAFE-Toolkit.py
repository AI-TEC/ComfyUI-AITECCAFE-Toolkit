import os
import configparser
import openai
from openai import OpenAI
import folder_paths

class ChatGPTNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True, "default": ""}),
                "role_setting": ("STRING", {"multiline": True, "default": "あなたは親切なアシスタントです。"}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_text",)
    FUNCTION = "generate_response"
    CATEGORY = "text"

    def __init__(self):
        self.last_response = ""

    def generate_response(self, input_text, role_setting, api_key):
        """ChatGPTでテキストを生成"""
        if not api_key.strip():
            print("APIキーが入力されていません")
            return ("",)
        
        if not input_text.strip():
            return ("",)
        
        try:
            # APIキーを使ってクライアントを初期化
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": role_setting},
                    {"role": "user", "content": input_text}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content
            self.last_response = generated_text
            return (generated_text,)
            
        except Exception as e:
            print(f"ChatGPT API呼び出しエラー: {e}")
            return ("",)

    @classmethod
    def IS_CHANGED(s, input_text, role_setting, api_key):
        # 入力が変わった場合に再実行
        return float("nan")


# ノードクラスマッピングに追加
NODE_CLASS_MAPPINGS = {
    "ChatGPTNode": ChatGPTNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatGPTNode": "AITEC ChatGPT Text Generator",
}

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

class CustomStringMergeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_string1": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "use_string2": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "use_string3": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
            },
            "optional": {  # string1～3をoptionalに移動
                "string1": ("STRING", {"default": ""}),
                "string2": ("STRING", {"default": ""}),
                "string3": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_string",)
    
    FUNCTION = "merge_strings"
    
    CATEGORY = "text"
    
    def merge_strings(self, string1="", string2="", string3="", use_string1=True, use_string2=True, use_string3=False):
        print(f"Switches: use_string1={use_string1}, use_string2={use_string2}, use_string3={use_string3}")
        strings = []
        if use_string1 and string1:
            strings.append(string1)
        if use_string2 and string2:
            strings.append(string2)
        if use_string3 and string3:
            strings.append(string3)
        
        merged = " \n".join(strings) if strings else ""
        return (merged,)

NODE_CLASS_MAPPINGS = {
    "CustomStringMerge": CustomStringMergeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomStringMerge": "AITEC Custom String Merge"
}

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

import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
from openai import OpenAI

class OpenAIImageModeration:
    
    #OpenAI omni-moderation-latestモデルを使用して画像のモデレーションを行うノード
    
    CATEGORY_NAMES_JA = {
        "harassment": "嫌がらせ",
        "harassment_threatening": "嫌がらせ/脅迫",
        "hate": "憎悪",
        "hate_threatening": "憎悪/脅迫",
        "illicit": "違法行為",
        "illicit_violent": "違法行為/暴力",
        "self_harm": "自傷",
        "self_harm_intent": "自傷/意図",
        "self_harm_instructions": "自傷/指示",
        "sexual": "性的内容",
        "sexual_minors": "性的内容/未成年",
        "violence": "暴力",
        "violence_graphic": "暴力/グラフィック"
    }
    
    CATEGORIES = list(CATEGORY_NAMES_JA.keys())
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "sk-proj-..."
                }),
                "output_format": (["詳細", "簡潔", "JSON"],),
                "language": (["日本語", "English"],),
                "block_flagged": ("BOOLEAN", {
                    "default": False,
                    "label_on": "ブロック",
                    "label_off": "常に出力"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "moderation_result")
    FUNCTION = "moderate_image"
    CATEGORY = "image/analysis"
    
    def tensor_to_base64(self, image_tensor):
        
        #ComfyUIの画像テンソル (B, H, W, C) をbase64エンコードされた文字列に変換
        
        # テンソルを numpy配列に変換 (0-1の範囲を0-255に変換)
        image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        # PIL Imageに変換
        pil_image = Image.fromarray(image_np)
        
        # バイトストリームに保存
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=95)
        
        # base64エンコード
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{image_base64}"
    
    def create_blank_image(self, original_image):
        
        #元の画像と同じサイズの黒い画像を作成
        
        # 元の画像と同じ形状のゼロテンソルを作成
        blank = torch.zeros_like(original_image)
        return blank
    
    def moderate_image(self, image, api_key, output_format="詳細", language="日本語", block_flagged=False):
        
        #画像をモデレーションAPIで分析
        
        try:
            # APIキーの検証
            if not api_key or api_key == "sk-proj-..." or len(api_key) < 20:
                error_msg = "エラー: 有効なOpenAI APIキーを入力してください。" if language == "日本語" else "Error: Please enter a valid OpenAI API key."
                return (image, error_msg)
            
            # OpenAIクライアントの初期化
            client = OpenAI(api_key=api_key)
            
            # 画像をbase64に変換
            base64_url = self.tensor_to_base64(image)
            
            # モデレーションAPIを呼び出し
            response = client.moderations.create(
                model="omni-moderation-latest",
                input=[{"type": "image_url", "image_url": {"url": base64_url}}]
            )
            
            result = response.results[0]
            
            # スコアを辞書形式で取得
            scores = {cat: float(getattr(result.category_scores, cat)) for cat in self.CATEGORIES}
            max_category = max(scores, key=scores.get)
            max_score = scores[max_category]
            
            # 不適切コンテンツが検出され、ブロックモードが有効な場合
            output_image = image
            if result.flagged and block_flagged:
                output_image = self.create_blank_image(image)
            
            # 出力フォーマットに応じて結果を整形
            if output_format == "JSON":
                output = json.dumps({
                    "flagged": result.flagged,
                    "blocked": result.flagged and block_flagged,
                    "max_category": max_category,
                    "max_score": max_score,
                    "scores": scores
                }, indent=2, ensure_ascii=False)
            
            elif output_format == "簡潔":
                if language == "日本語":
                    status = "検出された" if result.flagged else "検出されなかった"
                    category_name = self.CATEGORY_NAMES_JA[max_category]
                    output = f"不適切コンテンツ: {status}\n最高スコア: {category_name} ({max_score:.4f})"
                    if result.flagged and block_flagged:
                        output += "\n⚠️ 画像はブロックされました"
                else:
                    status = "FLAGGED" if result.flagged else "NOT FLAGGED"
                    output = f"Status: {status}\nTop Category: {max_category} ({max_score:.4f})"
                    if result.flagged and block_flagged:
                        output += "\n⚠️ Image blocked"
            
            else:  # 詳細
                if language == "日本語":
                    status = "検出された" if result.flagged else "検出されなかった"
                    output = f"=== モデレーション結果 ===\n"
                    output += f"不適切コンテンツ: {status}\n"
                    if result.flagged and block_flagged:
                        output += f"画像出力: ⚠️ ブロックされました\n"
                    output += f"\n最高スコアカテゴリ: {self.CATEGORY_NAMES_JA[max_category]} ({max_score:.4f})\n\n"
                    output += "全カテゴリスコア:\n"
                    for cat in self.CATEGORIES:
                        score = scores[cat]
                        bar = "█" * int(score * 50)
                        output += f"  {self.CATEGORY_NAMES_JA[cat]:20s}: {score:.4f} {bar}\n"
                else:
                    status = "FLAGGED" if result.flagged else "NOT FLAGGED"
                    output = f"=== Moderation Result ===\n"
                    output += f"Status: {status}\n"
                    if result.flagged and block_flagged:
                        output += f"Image Output: ⚠️ BLOCKED\n"
                    output += f"\nTop Category: {max_category} ({max_score:.4f})\n\n"
                    output += "All Category Scores:\n"
                    for cat in self.CATEGORIES:
                        score = scores[cat]
                        bar = "█" * int(score * 50)
                        output += f"  {cat:25s}: {score:.4f} {bar}\n"
            
            return (output_image, output)
        
        except Exception as e:
            error_msg = f"エラーが発生しました: {str(e)}" if language == "日本語" else f"Error: {str(e)}"
            return (image, error_msg)


# ノードマッピング
NODE_CLASS_MAPPINGS = {
    "OpenAIImageModeration": OpenAIImageModeration
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIImageModeration": "AITEC OpenAI Image Moderation"
}