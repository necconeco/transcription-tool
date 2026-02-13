"""
動画文字起こしツール
- 動画ファイルをアップロード → 文字起こし
- 日本語・英語対応
- テキスト + SRT形式で出力
"""

import streamlit as st
from faster_whisper import WhisperModel
import ffmpeg
import srt
import tempfile
import os
from datetime import timedelta
from pathlib import Path
import anthropic
import time

# ページ設定
st.set_page_config(
    page_title="動画文字起こしツール",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 動画文字起こしツール")
st.markdown("動画ファイルをアップロードして、日本語・英語の文字起こしができます。")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")

    model_option = st.selectbox(
        "Whisper モデル",
        ["tiny", "base", "small", "medium"],
        index=1,  # デフォルトは base
        help="tiny: 最速・低精度 / base: バランス型 / small: 高精度 / medium: 最高精度（時間かかる）"
    )

    language_option = st.radio(
        "言語",
        ["自動検出", "日本語", "英語"]
    )

    st.markdown("---")
    st.header("📝 字幕設定")
    max_chars = st.slider(
        "1字幕あたりの最大文字数",
        min_value=30,
        max_value=150,
        value=50,
        step=5,
        help="VREW用：50文字がおすすめ"
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
        help="英語の文字起こし結果を日本語に翻訳します（Claude API使用）"
    )

    if enable_translation:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Claude APIキーを入力してください"
        )
        st.caption("※ 環境変数 ANTHROPIC_API_KEY でも設定可能")


def extract_audio(video_path: str, audio_path: str) -> None:
    """動画から音声を抽出"""
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(
        stream.audio,
        audio_path,
        acodec="pcm_s16le",
        ar=16000,
        ac=1
    )
    ffmpeg.run(stream, overwrite_output=True, quiet=True)


def format_srt_timestamp(seconds: float) -> str:
    """秒数をSRT形式のタイムスタンプに変換 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def split_text_by_length(text: str, max_chars: int) -> list:
    """テキストを最大文字数で分割（単語の途中で切らない）"""
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_chars:
            current_chunk = current_chunk + " " + word if current_chunk else word
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text]


def create_srt(segments, max_chars: int = 100) -> str:
    """Whisper のセグメントから VREW互換SRT形式を生成（文字数制限付き）"""
    lines = []
    idx = 1

    for segment in segments:
        text = segment.text.strip()
        start_time = segment.start
        end_time = segment.end
        duration = end_time - start_time

        # テキストを分割
        chunks = split_text_by_length(text, max_chars)
        chunk_duration = duration / len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_start = start_time + (i * chunk_duration)
            chunk_end = start_time + ((i + 1) * chunk_duration)

            start_ts = format_srt_timestamp(chunk_start)
            end_ts = format_srt_timestamp(chunk_end)

            lines.append(str(idx))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(chunk)
            lines.append("")
            idx += 1

    return "\n".join(lines)


def format_timestamp(seconds: float) -> str:
    """秒数を HH:MM:SS 形式に変換"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def create_timestamped_text(segments) -> str:
    """タイムスタンプ付きテキストを生成"""
    lines = []
    for segment in segments:
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()
        lines.append(f"[{start} - {end}] {text}")
    return "\n".join(lines)


def translate_segments_with_claude(segments_list: list, api_key: str, progress_callback=None) -> list:
    """Claude APIを使って英語セグメントを日本語に翻訳"""
    client = anthropic.Anthropic(api_key=api_key)
    translated_texts = []

    total = len(segments_list)
    for i, segment in enumerate(segments_list):
        original_text = segment.text.strip()

        if progress_callback:
            progress_callback(i + 1, total, original_text[:30] + "..." if len(original_text) > 30 else original_text)

        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system="あなたはプロの翻訳者です。英語を自然な日本語に翻訳してください。字幕用なので簡潔に。翻訳結果のみを出力し、説明は不要です。",
                messages=[
                    {"role": "user", "content": f"次の英語を日本語に翻訳してください:\n{original_text}"}
                ]
            )
            translated_text = message.content[0].text.strip()
            translated_texts.append(translated_text)
        except anthropic.RateLimitError:
            time.sleep(2)
            try:
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system="あなたはプロの翻訳者です。英語を自然な日本語に翻訳してください。字幕用なので簡潔に。翻訳結果のみを出力し、説明は不要です。",
                    messages=[
                        {"role": "user", "content": f"次の英語を日本語に翻訳してください:\n{original_text}"}
                    ]
                )
                translated_text = message.content[0].text.strip()
                translated_texts.append(translated_text)
            except Exception as e:
                translated_texts.append(f"[翻訳エラー: {str(e)[:20]}]")
        except Exception as e:
            translated_texts.append(f"[翻訳エラー: {str(e)[:20]}]")

        time.sleep(0.1)

    return translated_texts


def create_bilingual_srt(segments, translated_texts: list) -> str:
    """英語（上）+ 日本語（下）のバイリンガルSRTを生成"""
    lines = []

    for idx, (segment, ja_text) in enumerate(zip(segments, translated_texts), 1):
        start_ts = format_srt_timestamp(segment.start)
        end_ts = format_srt_timestamp(segment.end)
        en_text = segment.text.strip()

        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(en_text)
        lines.append(ja_text)
        lines.append("")

    return "\n".join(lines)


# メインコンテンツ
uploaded_file = st.file_uploader(
    "動画ファイルをアップロード",
    type=["mp4", "mov", "avi", "mkv", "flv", "webm", "m4v"]
)

if uploaded_file:
    st.success(f"✅ ファイル選択済み: {uploaded_file.name}")

    # ファイルサイズ表示
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"📁 ファイルサイズ: {file_size_mb:.1f} MB")

    if st.button("🚀 文字起こし開始", type="primary"):
        # 言語マッピング
        language_map = {
            "自動検出": None,
            "日本語": "ja",
            "英語": "en"
        }
        selected_language = language_map[language_option]

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

            # ステップ2: モデルロード
            with st.spinner(f"🤖 Whisper モデル ({model_option}) をロード中..."):
                model = WhisperModel(model_option, device="cpu", compute_type="int8")
            st.success("✅ モデルロード完了")

            # ステップ3: 文字起こし
            with st.spinner("📝 文字起こし中... (動画の長さによって時間がかかります)"):
                segments, info = model.transcribe(
                    temp_audio_path,
                    language=selected_language,
                    beam_size=5
                )
                # セグメントをリストに変換（ジェネレータなので）
                segments_list = list(segments)

            # 検出言語を表示
            if selected_language is None:
                st.info(f"🌐 検出言語: {info.language} (確率: {info.language_probability:.1%})")

            st.success("✅ 文字起こし完了！")

            # 翻訳処理（有効な場合）
            translated_texts = None
            if enable_translation:
                actual_api_key = api_key if api_key else os.environ.get("ANTHROPIC_API_KEY")
                if actual_api_key:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(current, total, text):
                        progress_bar.progress(current / total)
                        status_text.text(f"🌐 翻訳中... ({current}/{total}): {text}")

                    with st.spinner("🌐 Claude で日本語に翻訳中..."):
                        translated_texts = translate_segments_with_claude(
                            segments_list,
                            actual_api_key,
                            progress_callback=update_progress
                        )

                    progress_bar.empty()
                    status_text.empty()
                    st.success("✅ 翻訳完了！")
                else:
                    st.warning("⚠️ APIキーが設定されていないため翻訳をスキップしました")

            # 全文テキスト作成
            full_text = "\n".join([seg.text.strip() for seg in segments_list])

            # SRT 作成（文字数制限付き）
            srt_content = create_srt(segments_list, max_chars)

            # 結果表示
            st.markdown("---")
            st.header("📄 結果")

            # タイムスタンプ付きテキスト作成
            timestamped_text = create_timestamped_text(segments_list)

            # タブの構成を翻訳の有無で変更
            if translated_texts:
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 テキスト", "⏱️ タイムスタンプ付き", "🎬 SRT字幕", "🌐 バイリンガルSRT", "📊 セグメント詳細"])
            else:
                tab1, tab2, tab3, tab4 = st.tabs(["📝 テキスト", "⏱️ タイムスタンプ付き", "🎬 SRT字幕", "📊 セグメント詳細"])

            with tab1:
                st.text_area("全文テキスト", value=full_text, height=400)
                st.download_button(
                    label="📥 テキストをダウンロード",
                    data=full_text,
                    file_name=f"{Path(uploaded_file.name).stem}_transcription.txt",
                    mime="text/plain"
                )

            with tab2:
                st.markdown("翻訳時に便利なタイムスタンプ付き形式です")
                st.text_area("タイムスタンプ付きテキスト", value=timestamped_text, height=400)
                st.download_button(
                    label="📥 タイムスタンプ付きをダウンロード",
                    data=timestamped_text,
                    file_name=f"{Path(uploaded_file.name).stem}_timestamped.txt",
                    mime="text/plain"
                )

            with tab3:
                st.markdown("**VREW対応** - そのままVREWにインポートできます")
                st.text_area("SRT形式", value=srt_content, height=400)
                st.download_button(
                    label="📥 SRT をダウンロード（VREW対応）",
                    data=srt_content.encode('utf-8'),
                    file_name=f"{Path(uploaded_file.name).stem}_subtitles.srt",
                    mime="text/plain"
                )

            # バイリンガルSRTタブ（翻訳有効時のみ）
            if translated_texts:
                with tab4:
                    bilingual_srt = create_bilingual_srt(segments_list, translated_texts)
                    st.markdown("**英語（上）+ 日本語（下）のバイリンガル字幕**")
                    st.markdown("VREW / VLC / YouTube 等で使用可能")
                    st.text_area("バイリンガルSRT", value=bilingual_srt, height=400)
                    st.download_button(
                        label="📥 バイリンガルSRT をダウンロード",
                        data=bilingual_srt.encode('utf-8'),
                        file_name=f"{Path(uploaded_file.name).stem}_bilingual.srt",
                        mime="text/plain"
                    )

                with tab5:
                    st.markdown(f"**セグメント数:** {len(segments_list)}")
                    for idx, (seg, ja_text) in enumerate(zip(segments_list, translated_texts)):
                        with st.expander(f"⏱️ {seg.start:.1f}s - {seg.end:.1f}s"):
                            st.write(f"**英語:** {seg.text.strip()}")
                            st.write(f"**日本語:** {ja_text}")
            else:
                with tab4:
                    st.markdown(f"**セグメント数:** {len(segments_list)}")
                    for seg in segments_list:
                        with st.expander(f"⏱️ {seg.start:.1f}s - {seg.end:.1f}s"):
                            st.write(f"**テキスト:** {seg.text.strip()}")

        except ffmpeg.Error as e:
            st.error(f"❌ 音声抽出エラー: FFmpeg がインストールされているか確認してください")
            st.code("brew install ffmpeg  # macOS の場合")

        except Exception as e:
            st.error(f"❌ エラーが発生しました: {e}")

        finally:
            # クリーンアップ
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

# フッター
st.markdown("---")
st.markdown("*Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper)*")
