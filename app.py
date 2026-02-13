"""
動画文字起こしツール - UI層
Streamlit によるユーザーインターフェース
"""

import streamlit as st

# ページ設定（最初に呼び出す必要あり）
st.set_page_config(
    page_title="動画文字起こしツール",
    page_icon="🎬",
    layout="wide",
)

import tempfile
import os
from pathlib import Path

import ffmpeg

from transcriber import (
    Language,
    extract_audio,
    load_whisper_model,
    transcribe_audio,
    translate_segments,
    create_srt,
    create_bilingual_srt,
    create_timestamped_text,
    create_full_text,
    get_api_key,
)


# =============================================================================
# キャッシュ
# =============================================================================

@st.cache_resource
def get_whisper_model(model_name: str):
    """Whisperモデルをキャッシュ付きでロード"""
    return load_whisper_model(model_name)


# =============================================================================
# 処理ロジック
# =============================================================================

def _process_transcription(
    uploaded_file,
    model_option: str,
    language_option: str,
    max_chars: int,
    enable_translation: bool,
    api_key_input: str,
) -> None:
    """文字起こし処理のメインフロー"""
    selected_language = Language.DISPLAY_MAP[language_option]

    # 一時ファイル作成
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        temp_audio_path = tmp_audio.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_video:
        temp_video_path = tmp_video.name
        tmp_video.write(uploaded_file.getbuffer())

    try:
        # ステップ1: 音声抽出
        with st.spinner("🔊 音声を抽出中..."):
            extract_audio(temp_video_path, temp_audio_path)
        st.success("✅ 音声抽出完了")

        # ステップ2: モデルロード（キャッシュ利用）
        with st.spinner(f"🤖 Whisper モデル ({model_option}) をロード中..."):
            model = get_whisper_model(model_option)
        st.success("✅ モデルロード完了")

        # ステップ3: 文字起こし
        with st.spinner("📝 文字起こし中... (動画の長さによって時間がかかります)"):
            result = transcribe_audio(model, temp_audio_path, selected_language)

        # 検出言語を表示
        if result.detected_language:
            st.info(f"🌐 検出言語: {result.detected_language} (確率: {result.language_probability:.1%})")

        st.success("✅ 文字起こし完了！")

        # 翻訳処理
        translated_texts = _handle_translation(
            segments=result.segments,
            enable_translation=enable_translation,
            api_key_input=api_key_input,
        )

        # 結果表示
        _display_results(
            segments=result.segments,
            translated_texts=translated_texts,
            max_chars=max_chars,
            filename=uploaded_file.name,
        )

    except ffmpeg.Error:
        st.error("❌ 音声抽出エラー: FFmpeg がインストールされているか確認してください")
        st.code("brew install ffmpeg  # macOS の場合")

    except Exception as e:
        st.error(f"❌ エラーが発生しました: {e}")

    finally:
        # クリーンアップ
        for path in [temp_audio_path, temp_video_path]:
            if os.path.exists(path):
                os.remove(path)


def _handle_translation(
    segments: list,
    enable_translation: bool,
    api_key_input: str,
) -> list[str] | None:
    """翻訳処理を実行"""
    if not enable_translation:
        return None

    api_key = get_api_key(api_key_input)
    if not api_key:
        st.warning("⚠️ APIキーが設定されていないため翻訳をスキップしました")
        return None

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(current: int, total: int, text: str) -> None:
        progress_bar.progress(current / total)
        status_text.text(f"🌐 翻訳中... ({current}/{total}): {text}")

    with st.spinner("🌐 Claude で日本語に翻訳中..."):
        translated_texts = translate_segments(
            segments=segments,
            api_key=api_key,
            progress_callback=update_progress,
        )

    progress_bar.empty()
    status_text.empty()
    st.success("✅ 翻訳完了！")

    return translated_texts


def _display_results(
    segments: list,
    translated_texts: list[str] | None,
    max_chars: int,
    filename: str,
) -> None:
    """結果を表示"""
    st.markdown("---")
    st.header("📄 結果")

    stem = Path(filename).stem
    full_text = create_full_text(segments)
    timestamped_text = create_timestamped_text(segments)
    srt_content = create_srt(segments, max_chars)

    # タブ構成
    if translated_texts:
        tabs = st.tabs(["📝 テキスト", "⏱️ タイムスタンプ付き", "🎬 SRT字幕", "🌐 バイリンガルSRT", "📊 セグメント詳細"])
    else:
        tabs = st.tabs(["📝 テキスト", "⏱️ タイムスタンプ付き", "🎬 SRT字幕", "📊 セグメント詳細"])

    # タブ1: テキスト
    with tabs[0]:
        st.text_area("全文テキスト", value=full_text, height=400)
        st.download_button(
            label="📥 テキストをダウンロード",
            data=full_text,
            file_name=f"{stem}_transcription.txt",
            mime="text/plain",
        )

    # タブ2: タイムスタンプ付き
    with tabs[1]:
        st.markdown("翻訳時に便利なタイムスタンプ付き形式です")
        st.text_area("タイムスタンプ付きテキスト", value=timestamped_text, height=400)
        st.download_button(
            label="📥 タイムスタンプ付きをダウンロード",
            data=timestamped_text,
            file_name=f"{stem}_timestamped.txt",
            mime="text/plain",
        )

    # タブ3: SRT
    with tabs[2]:
        st.markdown("**VREW対応** - そのままVREWにインポートできます")
        st.text_area("SRT形式", value=srt_content, height=400)
        st.download_button(
            label="📥 SRT をダウンロード（VREW対応）",
            data=srt_content.encode("utf-8"),
            file_name=f"{stem}_subtitles.srt",
            mime="text/plain",
        )

    # タブ4以降: 翻訳の有無で分岐
    if translated_texts:
        with tabs[3]:
            bilingual_srt = create_bilingual_srt(segments, translated_texts)
            st.markdown("**英語（上）+ 日本語（下）のバイリンガル字幕**")
            st.markdown("VREW / VLC / YouTube 等で使用可能")
            st.text_area("バイリンガルSRT", value=bilingual_srt, height=400)
            st.download_button(
                label="📥 バイリンガルSRT をダウンロード",
                data=bilingual_srt.encode("utf-8"),
                file_name=f"{stem}_bilingual.srt",
                mime="text/plain",
            )

        with tabs[4]:
            st.markdown(f"**セグメント数:** {len(segments)}")
            for seg, ja_text in zip(segments, translated_texts):
                with st.expander(f"⏱️ {seg.start:.1f}s - {seg.end:.1f}s"):
                    st.write(f"**英語:** {seg.text}")
                    st.write(f"**日本語:** {ja_text}")
    else:
        with tabs[3]:
            st.markdown(f"**セグメント数:** {len(segments)}")
            for seg in segments:
                with st.expander(f"⏱️ {seg.start:.1f}s - {seg.end:.1f}s"):
                    st.write(f"**テキスト:** {seg.text}")


# =============================================================================
# UI - ヘッダー
# =============================================================================

st.title("🎬 動画文字起こしツール")
st.markdown("動画ファイルをアップロードして、日本語・英語の文字起こしができます。")


# =============================================================================
# UI - サイドバー
# =============================================================================

with st.sidebar:
    st.header("⚙️ 設定")

    model_option = st.selectbox(
        "Whisper モデル",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="tiny: 最速・低精度 / base: バランス型 / small: 高精度 / medium: 最高精度（時間かかる）",
    )

    language_option = st.radio(
        "言語",
        list(Language.DISPLAY_MAP.keys()),
    )

    st.markdown("---")
    st.header("📝 字幕設定")

    max_chars = st.slider(
        "1字幕あたりの最大文字数",
        min_value=30,
        max_value=150,
        value=50,
        step=5,
        help="VREW用：50文字がおすすめ",
    )

    st.markdown("---")
    st.markdown("### モデル目安")
    st.markdown("""
    | モデル | 速度 | 精度 |
    |--------|------|------|
    | tiny | ⚡⚡⚡ | ⭐ |
    | base | ⚡⚡ | ⭐⭐ |
    | small | ⚡ | ⭐⭐⭐ |
    | medium | 🐢 | ⭐⭐⭐⭐ |
    """)

    st.markdown("---")
    st.header("🌐 翻訳設定")

    enable_translation = st.checkbox(
        "日本語翻訳を有効にする",
        value=False,
        help="英語の文字起こし結果を日本語に翻訳します（Claude API使用）",
    )

    api_key_input = ""
    if enable_translation:
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Claude APIキーを入力してください",
        )
        st.caption("※ 環境変数 ANTHROPIC_API_KEY でも設定可能")


# =============================================================================
# UI - メインコンテンツ
# =============================================================================

uploaded_file = st.file_uploader(
    "動画ファイルをアップロード",
    type=["mp4", "mov", "avi", "mkv", "flv", "webm", "m4v"],
)

if uploaded_file:
    st.success(f"✅ ファイル選択済み: {uploaded_file.name}")

    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"📁 ファイルサイズ: {file_size_mb:.1f} MB")

    if st.button("🚀 文字起こし開始", type="primary"):
        _process_transcription(
            uploaded_file=uploaded_file,
            model_option=model_option,
            language_option=language_option,
            max_chars=max_chars,
            enable_translation=enable_translation,
            api_key_input=api_key_input,
        )


# =============================================================================
# UI - フッター
# =============================================================================

st.markdown("---")
st.markdown("*Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)*")
