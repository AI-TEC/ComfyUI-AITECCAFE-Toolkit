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