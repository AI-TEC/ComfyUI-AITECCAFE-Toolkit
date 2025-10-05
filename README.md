# ComfyUI-AITEC-Nodes

このリポジトリには、AITECによって開発されたComfyUI用のカスタムノードが含まれています。

## インストール

1. ComfyUIの `custom_nodes` フォルダに移動します。
2. 以下のコマンドを使用して、このリポジトリをクローンします。
   ```bash
   git clone https://github.com/YOUR_USERNAME/ComfyUI-AITEC-Nodes.git
   ```
3. 必要な依存関係をインストールします。
   ```bash
   pip install -r ComfyUI-AITEC-Nodes/requirements.txt
   ```
4. ComfyUIを再起動します。

## ノード一覧

### AITEC ChatGPT Text Generator
ChatGPT APIを使用してテキストを生成します。

### AITEC Sequential Media Loader
指定されたフォルダから画像または動画のフレームをシーケンシャルにロードします。

### AITEC Custom String Merge
複数の文字列を結合します。

### AITEC Sequential Image Loader
指定されたフォルダから画像をシーケンシャルにロードします。

### AITEC OpenAI Image Moderation
OpenAIのモデレーションAPIを使用して画像を分析し、不適切なコンテンツを検出します。

## 依存関係

- `openai`
- `opencv-python`

## ライセンス

[LICENSE](LICENSE)ファイルを参照してください。

