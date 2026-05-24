# ComfyUI-AITECCAFE-Toolkit

This repository contains custom nodes for ComfyUI developed by AITECCAFE.  
日本語README → [README JA](README_JA.md)

## Installation

### ComfyUI Manager
1. Launch ComfyUI.
2. Open ComfyUI Manager.
3. Open Custom Nodes Manager.
4. Search for "AITECCAFE" and install ComfyUI_AITECCAFE_Toolkit.
    ![ComfyUI Manager](https://github.com/AI-TEC/images/blob/main/0001.jpg)
5. Restart ComfyUI.

### Command Line
1. Move to the `custom_nodes` folder in ComfyUI.
2. Clone this repository using the following command.
   ```bash
   git clone https://github.com/AI-TEC/ComfyUI-AITECCAFE-Toolkit
   ```
3. Install the required dependencies.
   ```bash
   pip install -r ComfyUI-AITECCAFE-Toolkit/requirements.txt
   ```
4. Restart ComfyUI.

### Installing llama-cpp-python
It is recommended to install llama-cpp-python from a `.whl` file that matches your environment.  
- Example: `.whl` file distribution source (by JamePeng)  
  https://github.com/JamePeng/llama-cpp-python

## Node List
<img src="https://github.com/AI-TEC/images/blob/main/0002.jpg" alt="Node List">
This Toolkit includes the following nodes.  

*   **💬 AITEC ChatGPT Chat**: Generates text using the OpenAI API.
*   **🛡️ AITEC Image Moderation**: Analyzes images using the OpenAI API and detects inappropriate content.
*   **🚫 AITEC NSFW Checker**: Analyzes images using opennsfw2 and detects inappropriate content.
*   **🖼️ AITEC Image Loader**: Loads images from a specified folder.
*   **🎞️ AITEC Media Loader**: Loads images or videos from a specified folder.
*   **🔗 AITEC String Merge**: Merges multiple strings.
*   **📦 AITEC LLM Loader**: Loads a local LLM model and allows it to be reused in Chat nodes.
*   **💬 AITEC LLM Chat**: Generates text using a local LLM.
*   **📦 AITEC LLM Vision Loader**: Loads an image-capable LLM model and mmproj for reuse in Vision nodes.
*   **🖼️ AITEC LLM Vision**: Performs image analysis and text generation using a local LLM. 
  
    ※ When using generation tasks and LLMs simultaneously, VRAM overflow may occur if there is not enough memory to load both models.
    
---

## 💬 AITEC ChatGPT Chat
<img src="https://github.com/AI-TEC/images/blob/main/0003.jpg" alt="ChatGPT Text Generator">
This node uses GPT-4.1 and outputs responses as text.  

**⚠️ Please handle your API key responsibly and use it at your own risk**  
**⚠️ If you distribute a workflow with the API key entered, others may be able to use your API key**  
**⚠️ Please manage your own credit usage**  

*   **input text**: Prompt sent to the ChatGPT API
*   **role setting**: System prompt sent to the ChatGPT API
*   **api_key**: OpenAI API key

---

## 🛡️ AITEC Image Moderation
<img src="https://github.com/AI-TEC/images/blob/main/0004.jpg" alt="OpenAI Image Moderation">
This node uses OpenAI's omni-moderation to detect inappropriate content and outputs the results as text.  

By enabling `block_flagged`, image output can be blocked when inappropriate content is detected.  
Please use the detected scores only as a rough reference.  
**⚠️ Please handle your API key responsibly and use it at your own risk**  
**⚠️ If you distribute a workflow with the API key entered, others may be able to use your API key**  
**⚠️ Please manage your own credit usage**  

*   **api_key**: OpenAI API key
*   **output_format**: Select result display format  detail/simple/json
*   **language**: Select language  English/Japanese
*   **block_flagged**: Block image output when inappropriate content is detected

---

*   ## 🚫 AITEC NSFW Checker
<img src="https://github.com/AI-TEC/images/blob/main/0008.jpg" alt="NSFW Checker">
This node uses opennsfw2 to detect inappropriate content and outputs the results as text.  

Unlike the Image Moderation node, no API key is required and it runs locally.  
Image output can be blocked when inappropriate content is detected.  
Please use the detected scores only as a rough reference.  

*   **block_nsfw**: Set whether to block NSFW content  pass through/block
*   **use_threshold**: Whether to block based on score threshold  enabled/disabled
*   **threshold**: Block when the score exceeds this value  
**The block_nsfw setting takes priority**　

---

## 🖼️ AITEC Image Loader
<img src="https://github.com/AI-TEC/images/blob/main/0007.jpg" alt="Sequential Image Loader">
This node can load images from a specified folder.  

*   **folder_path**: Path to the folder containing the images to load
*   **seed**: When set to increment, images in the folder can be loaded in filename order
*   **include_subfolders**: Set whether to load images from subfolders

---

## 🎞️ AITEC Media Loader
<img src="https://github.com/AI-TEC/images/blob/main/0006.jpg" alt="Sequential Media Loader">
This node can load media from a specified folder.  

*   **folder_path**: Path to the folder containing the media to load
*   **seed**: When set to increment, media in the folder can be loaded in filename order
*   **include_subfolders**: Set whether to load images from subfolders
*   **frame_index**: Set the starting frame for loading
*   **load_all_frames**: Set whether to load all frames
*   **max_frames**: Set the maximum number of frames to load
*   **frame_step**: Set the frame interval for loading

---

## 🔗 AITEC String Merge
<img src="https://github.com/AI-TEC/images/blob/main/0005.jpg" alt="Custom String Merge">
This node merges three strings in order from 1 to 3.  

*   **use_string1**: Set whether to use string1
*   **use_string2**: Set whether to use string2
*   **use_string3**: Set whether to use string3
*   **string1**: String 1 to use
*   **string2**: String 2 to use
*   **string3**: String 3 to use

---

## 📦 AITEC LLM Loader
<img src="https://github.com/AI-TEC/images/blob/main/0009.jpg" alt="AITEC LLM Loader">
Loads local LLM models (`.gguf`, `.safetensors`).  

Loaded models are shared across other LLM nodes for efficient memory usage.  

*   **model_file**: Model file to use (model location: `ComfyUI/models/llm`)
*   **n_ctx**: Context size (default: 4096; conversation history and think mode consume this heavily)
*   **n_gpu_layers**: Number of layers offloaded to GPU (`-1` for all)

Example: `.gguf` format is recommended when running alongside generation tasks (by HauhauCS)  
*   https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive
*   https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
  
    ※ When using generation tasks and LLMs simultaneously, VRAM overflow may occur if there is not enough memory to load both models.
    
---

## 💬 AITEC LLM Chat
<img src="https://github.com/AI-TEC/images/blob/main/0010.jpg" alt="AITEC LLM Chat">
Generates text using a local LLM.  

Since the model loaded by the Loader node is shared, memory usage does not increase even if multiple nodes are added.  

*   **model**: Model input from Loader
*   **system_prompt**: System prompt
*   **prompt**: Input text
*   **temperature**: Randomness of output
*   **top_p**: Probability control for token selection
*   **max_tokens**: Maximum number of generated tokens
*   **remove_think**: Remove `<think>` tags
*   **remove_chatml**: Clean up ChatML tags
*   **suppress_thinking**: Suppress reasoning process output
*   **reset_kv_cache**: Reset the KV cache before inference when ON (each inference becomes independent and prevents context exhaustion)
*   **unload_after_run**: When set to ON, the LLM model is unloaded after output (if you are referencing multiple deployments, see “How to unload LLM models” below)

- **Connection Example**: [AITEC LLM Loader] → MODEL → [AITEC LLM Chat]  
<img src="https://github.com/AI-TEC/images/blob/main/0013.jpg" alt="Connection example AITEC LLM">

---

## 📦 AITEC LLM Vision Loader
<img src="https://github.com/AI-TEC/images/blob/main/0011.jpg" alt="AITEC LLM Vision Loader">
Loads image-capable LLMs (Vision models) and mmproj files (`.gguf`, `.safetensors`).  

By combining with an mmproj file, image input becomes available.  

*   **model_file**: Model file to use (model location: `ComfyUI/models/llm`)
*   **mmproj_file**: Projection model for Vision (mmproj) (model location: `ComfyUI/models/llm`)
*   **n_ctx**: Context size (default: 4096; conversation history and think mode consume this heavily)
*   **n_gpu_layers**: GPU layer settings

Example: `.gguf` format is recommended when running alongside generation tasks (by HauhauCS)  
*   https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive
*   https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
  
    ※ When using generation tasks and LLMs simultaneously, VRAM overflow may occur if there is not enough memory to load both models.  
      In particular, LLM Vision is heavier than regular LLMs.
---

## 🖼️ AITEC LLM Vision
<img src="https://github.com/AI-TEC/images/blob/main/0012.jpg" alt="AITEC LLM Vision">
Performs analysis and description generation using images as input for the LLM.  

Supports up to 4 image inputs.  

*   **model**: Model input from Vision Loader
*   **system_prompt**: System prompt
*   **prompt**: Instruction text
*   **temperature**: Randomness of output
*   **top_p**: Probability control
*   **max_tokens**: Maximum number of tokens
*   **remove_think**: Remove `<think>` tags
*   **remove_chatml**: Clean up ChatML tags
*   **suppress_thinking**: Suppress reasoning
*   **unload_after_run**: When set to ON, the LLM model is unloaded after output (if you are referencing multiple deployments, see “How to unload LLM models” below)
*   **seed**: Used to force execution even when the input image and prompt have not changed (same seed does not guarantee identical output)
*   **control_after_generate**: Select anything other than fix to execute even when the input image and prompt have not changed
*   **image1〜image4**: Input images

- **Connection Example**: [AITEC LLM Vision Loader] → MODEL → [AITEC LLM Vision]  
<img src="https://github.com/AI-TEC/images/blob/main/0014.jpg" alt="Connection example AITEC LLM Vision">

---

## How to Use Model Unload with LLMs
If you deploy multiple LLM Chat or LLM Vision instances and want to unload models on each of them,  
please deploy multiple model loaders for each LLM.  
(If there is only one model loader, the process will continue without loading a model after the first unload.)

- **Connection Examples**:
<img src="https://github.com/AI-TEC/images/blob/main/0015.jpg" alt="Connection example multiple LLM Connect">

---

## Dependencies

- `openai`
- `opencv-python`
- `opennsfw2`
- `tensorflow`
- `llama-cpp-python`  
  `.whl` files are distributed by JamePeng.  
  https://github.com/JamePeng/llama-cpp-python

## License

MIT License  
Please refer to the [LICENSE](LICENSE) file.
