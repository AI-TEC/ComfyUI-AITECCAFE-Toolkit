import os
import configparser
import openai
from openai import OpenAI
import folder_paths
from PIL import Image
import torch
import numpy as np
import cv2
import base64
import io
import json

# ========================================
# ChatGPTNode
# ========================================
class ChatGPTNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True, "default": ""}),
                "role_setting": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_text",)
    FUNCTION = "generate_response"
    CATEGORY = "AITECCAFE-Toolkit"

    def __init__(self):
        self.last_response = ""

    def generate_response(self, input_text, role_setting, api_key):
        """Generate text with ChatGPT"""
        if not api_key.strip():
            print("API key not entered")
            return ("",)
        
        if not input_text.strip():
            return ("",)
        
        try:
            # Initialize the client with your API key
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
            print(f"ChatGPT API call error: {e}")
            return ("",)

    @classmethod
    def IS_CHANGED(s, input_text, role_setting, api_key):
        # Rerun if input changes
        return float("nan")


# ========================================
# SequentialMediaLoader
# ========================================
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
    CATEGORY = "AITECCAFE-Toolkit"

    def load_sequential_media(self, folder_path, seed, include_subfolders, frame_index, load_all_frames, max_frames, frame_step):
        # Valid extensions (images and videos)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        valid_extensions = image_extensions + video_extensions
        
        # Get Media File List
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

        # Sort files by name
        media_files.sort()
        
        # Use the seed as an index
        index = seed % len(media_files)
        selected_file = media_files[index]
        
        # Check the file extension
        file_ext = os.path.splitext(selected_file.lower())[1]
        filename = os.path.relpath(selected_file, folder_path)
        
        if file_ext in image_extensions:
            # Processing image files
            image, mask = self._load_image(selected_file)
            total_frames = 1
            print(f"Selected image: {filename} (Seed: {seed}, Index: {index})")
            
        elif file_ext in video_extensions:
            if load_all_frames:
                # Load all frames of the video
                image, mask, total_frames = self._load_all_video_frames(selected_file, max_frames, frame_step)
                print(f"Selected video (all frames): {filename} (Seed: {seed}, Index: {index}, Loaded frames: {image.shape[0]}/{total_frames})")
            else:
                # Import Single Frame
                image, mask, total_frames = self._load_video_frame(selected_file, frame_index)
                print(f"Selected video (single frame): {filename} (Seed: {seed}, Index: {index}, Frame: {frame_index}/{total_frames-1})")
        
        return (image, mask, filename, total_frames)
    
    def _load_image(self, image_path):
        """Load an image file"""
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        # Generate mask (grayscale conversion)
        mask = Image.open(image_path).convert('L')
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)[None,]
        
        return image_tensor, mask_tensor
    
    def _load_video_frame(self, video_path, frame_index):
        """Load a specific frame from a video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Adjust frame index within range整
        frame_index = frame_index % total_frames
        
        # Go to specified frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_index} from video: {video_path}")
        
        # BGR to RGB conversion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to tensor
        image_np = frame_rgb.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        # Generate mask (grayscale conversion)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_np = frame_gray.astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)[None,]
        
        return image_tensor, mask_tensor, total_frames
    
    def _load_all_video_frames(self, video_path, max_frames, frame_step):
        """Load all frames from a video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get the total number of frames
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
            
            # Skip frames according to frame_step
            if current_frame % frame_step == 0:
                # BGR to RGB conversion
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_np = frame_rgb.astype(np.float32) / 255.0
                frames.append(frame_np)
                
                # Generate mask (grayscale conversion)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask_np = frame_gray.astype(np.float32) / 255.0
                masks.append(mask_np)
                
                frame_count += 1
                
                # Stop when maximum number of frames is reached
                if frame_count >= max_frames:
                    break
            
            current_frame += 1
        
        cap.release()
        
        if not frames:
            raise ValueError(f"Could not load any frames from video: {video_path}")
        
        # Convert a list to a tensor
        image_tensor = torch.from_numpy(np.array(frames))
        mask_tensor = torch.from_numpy(np.array(masks))
        
        print(f"Loaded {len(frames)} frames from video (step: {frame_step})")
        
        return image_tensor, mask_tensor, total_frames


# ========================================
# CustomStringMergeNode
# ========================================
class CustomStringMergeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_string1": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "use_string2": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "use_string3": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
            },
            "optional": {  # string1~3をoptionalに移動
                "string1": ("STRING", {"default": ""}),
                "string2": ("STRING", {"default": ""}),
                "string3": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_string",)
    
    FUNCTION = "merge_strings"
    
    CATEGORY = "AITECCAFE-Toolkit"
    
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


# ========================================
# SequentialImageLoader
# ========================================
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
    CATEGORY = "AITECCAFE-Toolkit"

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
        index = seed % len(image_files) 
        
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


# ========================================
# OpenAIImageModeration
# ========================================
class OpenAIImageModeration:
    
    #A node that performs image moderation using the OpenAI omni-moderation-latest model.
    
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
                "output_format": (["detail", "simple", "JSON"],),
                "language": (["English", "Japanese"],),
                "block_flagged": ("BOOLEAN", {
                    "default": False,
                    "label_on": "block",
                    "label_off": "always output"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "moderation_result")
    FUNCTION = "moderate_image"
    CATEGORY = "AITECCAFE-Toolkit"
    
    def tensor_to_base64(self, image_tensor):
        
        #Convert ComfyUI image tensor (B, H, W, C) to a base64 encoded string
        
        # Convert tensor to numpy array (0-1 range converted to 0-255)
        image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        # PIL Convert to Image
        pil_image = Image.fromarray(image_np)
        
        # Save to byte stream
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=95)
        
        # Base64 encoding
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{image_base64}"
    
    def create_blank_image(self, original_image):
        
        #Create a black image the same size as the original image
        
        #Create a zero tensor with the same shape as the original image
        blank = torch.zeros_like(original_image)
        return blank
    
    def moderate_image(self, image, api_key, output_format="detail", language="Japanese", block_flagged=False):
        
        #Analyze images with our moderation API
        
        try:
            # API Key Validation
            if not api_key or api_key == "sk-proj-..." or len(api_key) < 20:
                error_msg = "Error: Please enter a valid OpenAI API key." if language == "Japanese" else "Error: Please enter a valid OpenAI API key."
                return (image, error_msg)
            
            # Initializing the OpenAI client
            client = OpenAI(api_key=api_key)
            
            # Convert image to base64
            base64_url = self.tensor_to_base64(image)
            
            # Call the moderation API
            response = client.moderations.create(
                model="omni-moderation-latest",
                input=[{"type": "image_url", "image_url": {"url": base64_url}}]
            )
            
            result = response.results[0]
            
            # Get the score in dictionary format
            scores = {cat: float(getattr(result.category_scores, cat)) for cat in self.CATEGORIES}
            max_category = max(scores, key=scores.get)
            max_score = scores[max_category]
            
            # Inappropriate content is detected and blocking mode is enabled
            output_image = image
            if result.flagged and block_flagged:
                output_image = self.create_blank_image(image)
            
            #Arrange the results according to the output format
            if output_format == "JSON":
                output = json.dumps({
                    "flagged": result.flagged,
                    "blocked": result.flagged and block_flagged,
                    "max_category": max_category,
                    "max_score": max_score,
                    "scores": scores
                }, indent=2, ensure_ascii=False)
            
            elif output_format == "simple":
                if language == "Japanese":
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
                if language == "Japanese":
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
            error_msg = f"An error has occurred: {str(e)}" if language == "Japanese" else f"Error: {str(e)}"
            return (image, error_msg)


# ========================================
# ノードマッピング(統合版)
# ========================================
NODE_CLASS_MAPPINGS = {
    "ChatGPTNode": ChatGPTNode,
    "SequentialMediaLoader": SequentialMediaLoader,
    "CustomStringMerge": CustomStringMergeNode,
    "SequentialImageLoader": SequentialImageLoader,
    "OpenAIImageModeration": OpenAIImageModeration
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatGPTNode": "ChatGPT Text Generator",
    "SequentialMediaLoader": "Sequential Media Loader",
    "CustomStringMerge": "Custom String Merge",
    "SequentialImageLoader": "Sequential Image Loader",
    "OpenAIImageModeration": "OpenAI Image Moderation"
}