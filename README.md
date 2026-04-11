# ComfyUI-AITECCAFE-Toolkit

このリポジトリには、AITECCAFEによって開発されたComfyUI用のカスタムノードが含まれています。

## note紹介
[noteに投稿したノードの紹介文です](https://note.com/ai_tec/n/ne3d398fe9548)

## インストール

### ComfyUI Manager
1. ComfyUIを起動します。
2. ComfyUI Managerを開きます。
3. Custom Nodes Managerを開きます。
4. "AITECCAFE"で検索するなどして、ComfyUI_AITECCAFE_Toolkitをインストールします。
    ![ComfyUI Manager](https://github.com/AI-TEC/images/blob/main/0001.jpg)
5. ComfyUIを再起動します。

### コマンドライン
1. ComfyUIの `custom_nodes` フォルダに移動します。
2. 以下のコマンドを使用して、このリポジトリをクローンします。
   ```bash
   git clone https://github.com/AI-TEC/ComfyUI-AITECCAFE-Toolkit
   ```
3. 必要な依存関係をインストールします。
   ```bash
   pip install -r ComfyUI-AITECCAFE-Toolkit/requirements.txt
   ```
4. ComfyUIを再起動します。

### llama-cpp-python インストール
- .whlファイルはJamePengさんが配布されています
  https://github.com/JamePeng/llama-cpp-python

## ノード一覧
<img src="https://github.com/AI-TEC/images/blob/main/0002.jpg" alt="Node List">
このToolkitには以下のノードが含まれています。

*   **💬 AITEC ChatGPT Chat**: ChatGPT APIを使用してテキストを生成します。
*   **🛡️ AITEC Image Moderation**: OpenAIのモデレーションAPIを使用して画像を分析し、不適切なコンテンツを検出します。
*   **🚫 AITEC NSFW Checker**: opennsfw2を使用して画像を分析し、不適切なコンテンツを検出します。
*   **🖼️ AITEC Image Loader**: 指定されたフォルダから画像をロードします。
*   **🎞️ AITEC Media Loader**: 指定されたフォルダから画像または動画をロードします。
*   **🔗 AITEC String Merge**: 複数の文字列を結合します。
*   **📦 AITEC LLM Loader**: ローカルLLMモデル（.gguf）を読み込み、Chatノードで再利用できるようにします。
*   **💬 AITEC LLM Chat**: ローカルLLMを使用してテキスト生成を行います。
*   **📦 AITEC LLM Vision Loader**: 画像対応LLMモデルとmmprojを読み込み、Visionノードで再利用できるようにします。
*   **🖼️ AITEC LLM Vision**: ローカルLLMを使用して画像解析・テキスト生成を行います。

---

## 💬 AITEC ChatGPT Chat
<img src="https://github.com/AI-TEC/images/blob/main/0003.jpg" alt="ChatGPT Text Generator">
このノードはGPT-4.1を利用して、回答がtextで出力されます。  

**⚠️APIキーは各自の責任で取り扱いに注意してご利用ください**  
**⚠️APIキーを入力した状態でワークフローを配布すると、他人がAPIキーを利用できる状態になります**  

*   **input text**: ChatGPT APIへ送るプロンプト
*   **role setting**: ChatGPT APIへ送るシステムプロンプト
*   **api_key**: OpenAIのAPIキー

---

## 🛡️ AITEC Image Moderation
<img src="https://github.com/AI-TEC/images/blob/main/0004.jpg" alt="OpenAI Image Moderation">
このノードはOpenAIのomni-moderationを利用して、不適切コンテンツを検出しtextで出力します。  

block_flaggedを設定することで、不適切コンテンツが検出された場合にimage出力をブロックすることができます。  
検出されるスコアは目安として参考にしてください。  
**⚠️APIキーは各自の責任で取り扱いに注意してご利用ください**  
**⚠️APIキーを入力した状態でワークフローを配布すると、他人がAPIキーを利用できる状態になります**  

*   **api_key**: OpenAIのAPIキー
*   **output_format**: 結果の表示形式を選択  detail/simple/json
*   **language**: 言語の選択を行います  English/Japanese
*   **block_flagged**: 不適切コンテンツが検出された場合image出力をブロックする

---

*   ## 🚫 AITEC NSFW Checker
<img src="https://github.com/AI-TEC/images/blob/main/0008.jpg" alt="NSFW Checker">
このノードはopennsfw2を利用して、不適切コンテンツを検出しtextで出力します。  

こちらのノードはAPIキーは不要で、ローカルで動作します。  
不適切コンテンツが検出された場合にimage出力をブロックすることができます。  
検出されるスコアは目安として参考にしてください。  

*   **block_nsfw**: NSFWをブロックするかどうか設定　pass through/block
*   **use_threshold**: score基準でブロックするかどうか  enabled/disabled
*   **threshold**: この値を超えた場合ブロックする  
**block_nsfwの設定が優先されます**　

---

## 🖼️ AITEC Image Loader
<img src="https://github.com/AI-TEC/images/blob/main/0007.jpg" alt="Sequential Image Loader">
このノードは指定されたフォルダ内の画像をロードすることができます

*   **folder_path**: 読み取りたい画像のあるフォルダのパス
*   **seed**: incrementを指定するとフォルダ内の画像をファイル名順に読み出すことができる
*   **include_subfolders**: サブフォルダの画像も読み出すかどうかを設定

---

## 🎞️ AITEC Media Loader
<img src="https://github.com/AI-TEC/images/blob/main/0006.jpg" alt="Sequential Media Loader">
このノードは指定されたフォルダ内のメディアをロードすることができます

*   **folder_path**: 読み取りたいメディアのあるフォルダのパス
*   **seed**: incrementを指定するとフォルダ内のメディアをファイル名順に読み出すことができる
*   **include_subfolders**: サブフォルダの画像も読み出すかどうかを設定
*   **frame_index**: 読み込みの開始フレームを設定
*   **load_all_frames**: すべてのフレームを読み込むかを設定
*   **max_frames**: 最大何フレームまで読み込むかを設定
*   **frame_step**: 何フレームごとに読み込むかを設定

---

## 🔗 AITEC String Merge
<img src="https://github.com/AI-TEC/images/blob/main/0005.jpg" alt="Custom String Merge">
このノードは3つのStringを１番から順にマージします

*   **use_string1**: string1を利用するかを設定
*   **use_string2**: string2を利用するかを設定
*   **use_string3**: string3を利用するかを設定
*   **string1**: 使用する文字列1
*   **string2**: 使用する文字列2
*   **string3**: 使用する文字列3

---

## 📦 AITEC LLM Loader
<img src="https://github.com/AI-TEC/images/blob/main/0009.jpg" alt="AITEC LLM Loader">
ローカル環境のLLM（.gguf）モデルを読み込みます。  
読み込まれたモデルは他のLLMノードで共有され、メモリ効率よく利用されます。

*   **model_file**: 使用するモデルファイル（モデルの保存先:ComfyUI/models/llm）
*   **n_ctx**: コンテキストサイズ
*   **n_gpu_layers**: GPUにオフロードするレイヤー数（-1で全て）

---

## 💬 AITEC LLM Chat
<img src="https://github.com/AI-TEC/images/blob/main/0010.jpg" alt="AITEC LLM Chat">
ローカルLLMを使用してテキスト生成を行います。  
Loaderノードで読み込んだモデルを共有して使用するため、複数配置してもメモリ使用量は増えません。

*   **model**: Loaderからのモデル入力
*   **system_prompt**: システムプロンプト
*   **prompt**: 入力テキスト
*   **temperature**: 出力のランダム性
*   **top_p**: トークン選択の確率制御
*   **max_tokens**: 最大生成トークン数
*   **remove_think**: <think>タグの削除
*   **remove_chatml**: ChatMLタグの整理
*   **suppress_thinking**: 推論過程の出力抑制

- **接続例**:[AITEC LLM Loader]        → MODEL → [AITEC LLM Chat] 
<img src="https://github.com/AI-TEC/images/blob/main/0013.jpg" alt="Connection example AITEC LLM">

---

## 📦 AITEC LLM Vision Loader
<img src="https://github.com/AI-TEC/images/blob/main/0011.jpg" alt="AITEC LLM Vision Loader">
画像対応LLM（Visionモデル）を読み込みます。  
mmprojファイルと組み合わせて、画像入力を扱えるようにします。

*   **model_file**: 使用するモデルファイル（モデルの保存先:ComfyUI/models/llm）
*   **mmproj_file**: Vision用プロジェクションモデル（mmproj）（モデルの保存先:ComfyUI/models/llm）
*   **n_ctx**: コンテキストサイズ
*   **n_gpu_layers**: GPUレイヤー設定

---

## 🖼️ AITEC LLM Vision
<img src="https://github.com/AI-TEC/images/blob/main/0012.jpg" alt="AITEC LLM Vision">
画像を入力としてLLMによる解析・説明生成を行います。  
最大4枚までの画像入力に対応しています。

*   **model**: Vision Loaderからのモデル入力
*   **system_prompt**: システムプロンプト
*   **prompt**: 指示文
*   **temperature**: 出力のランダム性
*   **top_p**: 確率制御
*   **max_tokens**: 最大トークン数
*   **remove_think**: <think>タグの削除
*   **remove_chatml**: ChatMLタグの整理
*   **suppress_thinking**: 推論抑制
*   **image1〜image4**: 入力画像

- **接続例**:[AITEC LLM Vision Loader] → MODEL → [AITEC LLM Vision]
<img src="https://github.com/AI-TEC/images/blob/main/0014.jpg" alt="Connection example AITEC LLM Vision">

## 依存関係

- `openai`
- `opencv-python`
- `opennsfw2`
- `tensorflow`
- `llama-cpp-python`
  .whlファイルはJamePengさんが配布されています。
  https://github.com/JamePeng/llama-cpp-python

## ライセンス

MIT License
[LICENSE](LICENSE)ファイルを参照してください。

