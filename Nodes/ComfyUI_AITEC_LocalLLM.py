# AITEC_LocalLLM.py
# ComfyUI custom node - Local LLM via llama-cpp-python
# Place this file directly in ComfyUI/custom_nodes/
#
# Model placement:
#   Main model  : ComfyUI/models/llm/*.gguf
#   MMProj model: ComfyUI/models/llm/*.gguf  (for vision node)
#
# Node structure:
#   [AITEC LLM Loader]        → MODEL ──→ [AITEC LLM Chat]    → text
#   [AITEC LLM Vision Loader] → MODEL ──→ [AITEC LLM Vision]  → text

import gc
import re
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# ComfyUI path helper
# ---------------------------------------------------------------------------
try:
    import folder_paths
    def _get_llm_model_dir() -> Path:
        try:
            dirs = folder_paths.get_folder_paths("llm")
            if dirs:
                return Path(dirs[0])
        except Exception:
            pass
        base = Path(getattr(folder_paths, "base_path", Path(__file__).parent.parent.parent))
        for name in ("llm", "LLM"):
            p = base / "models" / name
            if p.exists():
                return p
        return base / "models" / "llm"
except ImportError:
    folder_paths = None
    def _get_llm_model_dir() -> Path:
        return Path(__file__).parent.parent.parent / "models" / "llm"


def _scan_model_files() -> list:
    model_dir = _get_llm_model_dir()
    if not model_dir.exists():
        return ["(no models found)"]
    seen = set()
    files = []
    for ext in ("*.gguf", "*.GGUF", "*.safetensors"):
        for f in sorted(model_dir.glob(ext)):
            if f.name not in seen:
                seen.add(f.name)
                files.append(f.name)
    return files if files else ["(no models found)"]


def _resolve_model_path(filename: str) -> Path:
    return _get_llm_model_dir() / filename


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _remove_think_tags(text: str) -> str:
    """Remove the <think>...</think> tags output by Thinking models such as Qwen3 and Gemma4."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _remove_chatml_tags(text: str) -> str:
    """
    Handles the repeated output of <|im_start|>assistant～<|im_end|>, which occurs when combining chat_format="chatml" with Gemma4, etc.

    Specifications:
      - If there are multiple <|im_start|>assistant ... <|im_end|> blocks, return only the contents of the first block
      - If no blocks are found, return the content as-is (without affecting other models)
      - Also remove any standalone <|im_start|> / <|im_end|> tags
    """
    m = re.search(r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    cleaned = re.sub(r"<\|im_start\|>\w*\s*", "", text)
    cleaned = re.sub(r"<\|im_end\|>", "", cleaned)
    return cleaned.strip()


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    t = tensor[0] if tensor.ndim == 4 else tensor
    arr = (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _pil_to_base64_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_status(finish_reason: str) -> str:
    if finish_reason == "length":
        return "ok (finish_reason=length: The max_tokens limit has been reached. If you increase this value, the full text will be displayed.)"
    return "ok"


# Instructions to add to the system prompt when in inference suppression mode
_NOTHINK_INSTRUCTIONS = (
    "---\n"
    "You must answer directly without any internal thinking, reasoning, or step-by-step process.\n"
    "suppress the output of the same answer repeatedly.\n"
    "/set nothink\n"
    "/nothink"
)

def _apply_nothink(system_prompt: str, suppress_thinking: bool) -> str:
    """When `suppress_thinking=True`, add an instruction to suppress reasoning to the system prompt."""
    if not suppress_thinking:
        return system_prompt
    base = system_prompt.strip()
    if base:
        return base + "\n" + _NOTHINK_INSTRUCTIONS
    return _NOTHINK_INSTRUCTIONS


def _clean_output(text: str, remove_think: bool, remove_chatml: bool) -> str:
    """Apply all post-processing steps at once. Order: Remove CHATML → Remove THINK"""
    if remove_chatml:
        text = _remove_chatml_tags(text)
    if remove_think:
        text = _remove_think_tags(text)
    return text


# ---------------------------------------------------------------------------
# LLM model wrapper  (passed between Loader → Inference nodes)
# ---------------------------------------------------------------------------

class LLMModel:
    """A wrapper that stores a loaded model. It is passed between nodes via the MODEL pin."""
    def __init__(self, llm, model_file: str, has_vision: bool = False):
        self.llm = llm
        self.model_file = model_file
        self.has_vision = has_vision

    def __repr__(self):
        kind = "Vision" if self.has_vision else "Chat"
        return f"<LLMModel [{kind}] {self.model_file}>"


# ---------------------------------------------------------------------------
# Node 1: AITEC_LLM_Loader  (LLM Loader)
# ---------------------------------------------------------------------------

class AITEC_LLM_Loader:
    """
    Text-Only LLM Loader
    - Loads a model and outputs it to the MODEL pin
    - Can be shared among multiple Chat nodes
    """

    _cache: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        model_list = _scan_model_files()
        return {
            "required": {
                "model_file":   (model_list, {"default": model_list[0]}),
                "n_ctx":        ("INT",  {"default": 4096, "min": 512, "max": 131072, "step": 512,
                                          "tooltip": "Context window size (Qwen3思考モデルは16384以上推奨)"}),
                "n_gpu_layers": ("INT",  {"default": -1,   "min": -1,  "max": 200,   "step": 1,
                                          "tooltip": "-1 = Send all layers to the GPU"}),
            },
        }

    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "AITEC/LocalLLM"

    def _cache_key(self, model_file, n_ctx, n_gpu_layers):
        return f"chat|{model_file}|{n_ctx}|{n_gpu_layers}"

    def load(self, model_file: str, n_ctx: int, n_gpu_layers: int):
        if model_file == "(no models found)":
            raise ValueError("There are no model files in ComfyUI/models/llm/")

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("[AITEC] llama_cpp is not available. Please install llama-cpp-python.")

        key = self._cache_key(model_file, n_ctx, n_gpu_layers)
        if key in AITEC_LLM_Loader._cache:
            print(f"[AITEC_Loader] Load model from cache: {model_file}")
            return (AITEC_LLM_Loader._cache[key],)

        model_path = _resolve_model_path(model_file)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"[AITEC_Loader] Loading model: {model_path.name}")
        llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format="chatml",
            verbose=False,
        )
        wrapper = LLMModel(llm, model_file, has_vision=False)
        AITEC_LLM_Loader._cache[key] = wrapper
        print(f"[AITEC_Loader] Loading complete: {model_file}")
        return (wrapper,)


# ---------------------------------------------------------------------------
# Node 2: AITEC_LLM_Vision_Loader  (LLM & Vision Loader)
# ---------------------------------------------------------------------------

class AITEC_LLM_Vision_Loader:
    """
    Vision LLM Loader
    - Loads the main model and mmproj and outputs them to the MODEL pin
    - Can be shared across multiple Vision nodes
    """

    _cache: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        model_list = _scan_model_files()
        return {
            "required": {
                "model_file":   (model_list, {"default": model_list[0]}),
                "mmproj_file":  (model_list, {"default": model_list[0],
                                              "tooltip": "Vision projection model (mmproj-*.gguf)"}),
                "n_ctx":        ("INT",  {"default": 4096, "min": 512, "max": 131072, "step": 512}),
                "n_gpu_layers": ("INT",  {"default": -1,   "min": -1,  "max": 200,   "step": 1,
                                          "tooltip": "-1 = Send all layers to the GPU"}),
            },
        }

    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "AITEC/LocalLLM"

    def _cache_key(self, model_file, mmproj_file, n_ctx, n_gpu_layers):
        return f"vision|{model_file}|{mmproj_file}|{n_ctx}|{n_gpu_layers}"

    def load(self, model_file: str, mmproj_file: str, n_ctx: int, n_gpu_layers: int):
        if model_file == "(no models found)":
            raise ValueError("There are no model files in ComfyUI/models/llm/")

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("[AITEC] llama_cpp is not available. Please install llama-cpp-python.")

        key = self._cache_key(model_file, mmproj_file, n_ctx, n_gpu_layers)
        if key in AITEC_LLM_Vision_Loader._cache:
            print(f"[AITEC_VisionLoader] Load model from cache: {model_file}")
            return (AITEC_LLM_Vision_Loader._cache[key],)

        model_path  = _resolve_model_path(model_file)
        mmproj_path = _resolve_model_path(mmproj_file)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not mmproj_path.exists():
            raise FileNotFoundError(f"MMProj not found: {mmproj_path}")

        # Vision chat handler を探す
        chat_handler = None
        for handler_name in ("Qwen25VLChatHandler", "Qwen3VLChatHandler",
                             "MoondreamChatHandler", "LlavaImageChatHandler"):
            try:
                import llama_cpp.llama_chat_format as fmt
                h_cls = getattr(fmt, handler_name, None)
                if h_cls is not None:
                    chat_handler = h_cls(clip_model_path=str(mmproj_path), verbose=False)
                    print(f"[AITEC_VisionLoader] Chat Handler: {handler_name}")
                    break
            except Exception:
                pass

        if chat_handler is None:
            raise RuntimeError(
                "[AITEC_VisionLoader] Vision chat handler not found.\n"
                "Requires a vision-compatible version of llama-cpp-python (JamePeng fork recommended):\n"
                "  https://github.com/JamePeng/llama-cpp-python/releases"
            )

        print(f"[AITEC_VisionLoader] Loading model: {model_path.name}")
        llm = Llama(
            model_path=str(model_path),
            chat_handler=chat_handler,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        wrapper = LLMModel(llm, model_file, has_vision=True)
        AITEC_LLM_Vision_Loader._cache[key] = wrapper
        print(f"[AITEC_VisionLoader] Loading model: {model_file}")
        return (wrapper,)


# ---------------------------------------------------------------------------
# Node 3: AITEC_LLM_Chat  (Text Inference)
# ---------------------------------------------------------------------------

class AITEC_LLM_Chat:
    """
    Text Inference Node
    - Receives the MODEL from the Loader and performs inference
    - Even if multiple nodes are deployed, they share the MODEL, so there is no increase in memory usage
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("LLM_MODEL", {}),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                }),
                "prompt": ("STRING", {
                    "default": "Write a creative image generation prompt.",
                    "multiline": True,
                }),
                "temperature":   ("FLOAT",   {"default": 0.7,  "min": 0.0, "max": 2.0,   "step": 0.05}),
                "top_p":         ("FLOAT",   {"default": 0.95, "min": 0.0, "max": 1.0,   "step": 0.01}),
                "max_tokens":    ("INT",     {"default": 4096, "min": 64,  "max": 32768, "step": 64,
                                              "tooltip": "Thinking model with 4096 or more is recommended"}),
                "remove_think":  ("BOOLEAN", {"default": True,
                                              "tooltip": "Remove the <think>...</think> block (Qwen3, Gemma4, etc.)"}),
                "remove_chatml": ("BOOLEAN", {"default": True,
                                              "tooltip": "Remove repetitions of <|im_start|>assistant～<|im_end|> and return only the first response (Gemma4, etc.)"}),
                "suppress_thinking": ("BOOLEAN", {"default": False,
                                                   "tooltip": "When enabled, adds an inference suppression instruction to the system prompt (for Thinking models such as Qwen3 and Gemma4)"}),
                "reset_kv_cache":    ("BOOLEAN", {"default": True,
                                                   "tooltip": "推論前にKVキャッシュをリセット。ONでコンテキスト枯渇を防止（Qwen3等の思考モデルに推奨）。OFFで会話履歴を保持。"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "used_model", "status")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "AITEC/LocalLLM"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute (since NaN is not equal to itself, the cache is invalidated)
        return float("NaN")

    def run(self, model: LLMModel, system_prompt: str, prompt: str,
            temperature: float, top_p: float, max_tokens: int,
            remove_think: bool, remove_chatml: bool, suppress_thinking: bool,
            reset_kv_cache: bool = True):

        if not prompt.strip():
            return ("", "", "error: The prompt is empty")

        try:
            # KVキャッシュをリセットして毎回クリーンな状態で推論する
            # Qwen3等の思考モデルは<think>ブロックで大量トークンを消費するため
            # リセットしないと数回でコンテキスト枯渇し空レスポンスになる
            if reset_kv_cache:
                try:
                    model.llm.reset()
                except Exception:
                    pass

            messages = []
            final_system = _apply_nothink(system_prompt, suppress_thinking)
            if final_system:
                messages.append({"role": "system", "content": final_system})
            messages.append({"role": "user", "content": prompt.strip()})

            response = model.llm.create_chat_completion(
                messages=messages,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            )

            text = (response["choices"][0]["message"]["content"] or "").strip()
            finish = response["choices"][0].get("finish_reason", "")

            # finish_reasonが空の場合はコンテキスト枯渇の可能性を警告
            if not finish:
                finish_status = "warning: finish_reason is empty. Context may be exhausted. Try increasing n_ctx."
            else:
                finish_status = _make_status(finish)

            text = _clean_output(text, remove_think=remove_think, remove_chatml=remove_chatml)

            if not text:
                return ("", model.model_file, "warning: Empty response. Context may be exhausted. Try increasing n_ctx in Loader.")

            return (text, model.model_file, finish_status)

        except Exception as e:
            return ("", getattr(model, "model_file", ""), f"error: {e}")


# ---------------------------------------------------------------------------
# Node 4: AITEC_LLM_Vision  (Vision Inference)
# ---------------------------------------------------------------------------

class AITEC_LLM_Vision:
    """
    Vision Inference Node
    - Receives a MODEL from the Vision Loader and performs inference
    - Supports up to 4 image inputs
    - No increase in memory usage even when multiple nodes are deployed, as they share the same MODEL
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("LLM_MODEL", {}),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                }),
                "prompt": ("STRING", {
                    "default": "Describe the image(s) in detail.",
                    "multiline": True,
                }),
                "temperature":   ("FLOAT",   {"default": 0.7,  "min": 0.0, "max": 2.0,   "step": 0.05}),
                "top_p":         ("FLOAT",   {"default": 0.95, "min": 0.0, "max": 1.0,   "step": 0.01}),
                "max_tokens":    ("INT",     {"default": 4096, "min": 64,  "max": 32768, "step": 64,
                                              "tooltip": "Thinking model with 4096 or more is recommended"}),
                "remove_think":  ("BOOLEAN", {"default": True,
                                              "tooltip": "Remove the <think>...</think> block (Qwen3, Gemma4, etc.)"}),
                "remove_chatml": ("BOOLEAN", {"default": True,
                                              "tooltip": "Remove repetitions of <|im_start|>assistant～<|im_end|> and return only the first response (Gemma4, etc.)"}),
                "suppress_thinking": ("BOOLEAN", {"default": False,
                                                   "tooltip": "When enabled, adds an inference suppression instruction to the system prompt (for Thinking models such as Qwen3 and Gemma4)"}),
                "reset_kv_cache":    ("BOOLEAN", {"default": True,
                                                   "tooltip": "推論前にKVキャッシュをリセット。ONでコンテキスト枯渇を防止（Qwen3等の思考モデルに推奨）。OFFで会話履歴を保持。"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True,
                                 "tooltip": "実行のたびに値を変えることでキャッシュをスキップして強制再実行します。"}),
            },
            "optional": {
                "image1": ("IMAGE", {}),
                "image2": ("IMAGE", {}),
                "image3": ("IMAGE", {}),
                "image4": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "used_model", "status")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "AITEC/LocalLLM"

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return seed

    def run(self, model: LLMModel, system_prompt: str, prompt: str,
            temperature: float, top_p: float, max_tokens: int,
            remove_think: bool, remove_chatml: bool, suppress_thinking: bool,
            reset_kv_cache: bool = True, seed: int = 0,
            image1=None, image2=None, image3=None, image4=None):

        try:
            # Visionモデルのコンテキスト完全リセット
            # chat_handler経由の画像エンコードでn_past等の内部状態が残るため
            # reset()だけでは "Fatal Decode Error at Pos 0" が発生する
            if reset_kv_cache:
                try:
                    llm = model.llm
                    # llama_cpp内部APIでKVキャッシュを完全クリア
                    if hasattr(llm, '_ctx') and llm._ctx is not None:
                        import llama_cpp.llama_cpp as _lib
                        _lib.llama_kv_cache_clear(llm._ctx)
                    # n_tokensとn_pastをリセット
                    if hasattr(llm, 'n_tokens'):
                        llm.n_tokens = 0
                    if hasattr(llm, '_n_past'):
                        llm._n_past = 0
                    llm.reset()
                except Exception as e:
                    print(f"[AITEC_Vision] reset warning: {e}")

            messages = []
            final_system = _apply_nothink(system_prompt, suppress_thinking)
            if final_system:
                messages.append({"role": "system", "content": final_system})

            user_content = []
            for img_tensor in [image1, image2, image3, image4]:
                if img_tensor is not None:
                    b64 = _pil_to_base64_png(_tensor_to_pil(img_tensor))
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
            user_content.append({"type": "text", "text": prompt.strip()})
            messages.append({"role": "user", "content": user_content})

            response = model.llm.create_chat_completion(
                messages=messages,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            )

            text = (response["choices"][0]["message"]["content"] or "").strip()
            finish = response["choices"][0].get("finish_reason", "")

            if not finish:
                finish_status = "warning: finish_reason is empty. Context may be exhausted. Try increasing n_ctx."
            else:
                finish_status = _make_status(finish)

            text = _clean_output(text, remove_think=remove_think, remove_chatml=remove_chatml)

            if not text:
                return ("", model.model_file, "warning: Empty response. Context may be exhausted. Try increasing n_ctx in Loader.")

            return (text, model.model_file, finish_status)

        except Exception as e:
            return ("", getattr(model, "model_file", ""), f"error: {e}")


# ---------------------------------------------------------------------------
# ComfyUI node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "AITEC_LLM_Loader":        AITEC_LLM_Loader,
    "AITEC_LLM_Vision_Loader": AITEC_LLM_Vision_Loader,
    "AITEC_LLM_Chat":          AITEC_LLM_Chat,
    "AITEC_LLM_Vision":        AITEC_LLM_Vision,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AITEC_LLM_Loader":        "📦 AITEC LLM Loader",
    "AITEC_LLM_Vision_Loader": "📦 AITEC LLM Vision Loader",
    "AITEC_LLM_Chat":          "💬 AITEC LLM Chat",
    "AITEC_LLM_Vision":        "🖼️ AITEC LLM Vision",
}
