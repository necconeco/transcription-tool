# 動画文字起こしツール

動画ファイルをアップロードして、日本語・英語の文字起こしができるWebアプリです。

## 機能

- 動画ファイル（mp4, mov, avi, mkv など）をアップロード
- 日本語・英語の文字起こし（自動検出も可能）
- テキスト形式 + SRT字幕形式で出力
- タイムスタンプ付きセグメント表示

## セットアップ

### 1. FFmpeg のインストール

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

### 2. Python 依存関係のインストール

```bash
cd 文字起こしツール
pip install -r requirements.txt
```

### 3. アプリ起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

## 使い方

1. サイドバーでモデルと言語を選択
2. 動画ファイルをアップロード
3. 「文字起こし開始」ボタンをクリック
4. 結果をテキストまたはSRT形式でダウンロード

## モデル選択の目安

| モデル | 速度 | 精度 | 用途 |
|--------|------|------|------|
| tiny | 最速 | 低 | 概要把握、テスト用 |
| base | 速い | 中 | 一般的な用途（推奨） |
| small | 普通 | 高 | 正確さ重視 |
| medium | 遅い | 最高 | 専門用語・固有名詞が多い場合 |

## 技術スタック

- **フロントエンド**: Streamlit
- **文字起こし**: faster-whisper
- **音声抽出**: FFmpeg
- **字幕生成**: srt
