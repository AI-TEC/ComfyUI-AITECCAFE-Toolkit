# ComfyUI-AITECCAFE-Toolkit

このリポジトリには、AITECCAFEによって開発されたComfyUI用のカスタムノードが含まれています。

## インストール

1. ComfyUIの `custom_nodes` フォルダに移動します。
2. 以下のコマンドを使用して、このリポジトリをクローンします。
   ```bash
   git clone https://github.com/YOUR_USERNAME/ComfyUI-AITECCAFE-Toolkit.git
   ```
3. 必要な依存関係をインストールします。
   ```bash
   pip install -r ComfyUI-AITECCAFE-Toolkit/requirements.txt
   ```
4. ComfyUIを再起動します。

## ノード一覧

このToolkitには以下のノードが含まれています。

*   **AITECCAFE ChatGPT Text Generator**: ChatGPT APIを使用してテキストを生成します。
*   **AITECCAFE Sequential Media Loader**: 指定されたフォルダから画像または動画のフレームをシーケンシャルにロードします。
*   **AITECCAFE Custom String Merge**: 複数の文字列を結合します。
*   **AITECCAFE Sequential Image Loader**: 指定されたフォルダから画像をシーケンシャルにロードします。
*   **AITECCAFE OpenAI Image Moderation**: OpenAIのモデレーションAPIを使用して画像を分析し、不適切なコンテンツを検出します。

## 依存関係

- `openai`
- `opencv-python`

## ライセンス

[LICENSE](LICENSE)ファイルを参照してください。

