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