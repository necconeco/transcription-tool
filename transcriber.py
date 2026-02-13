"""
文字起こし・翻訳のコアロジック
UI層から独立した純粋なビジネスロジック
"""

import time
import os
from dataclasses import dataclass
from typing import Callable, Optional

import ffmpeg
import anthropic
from faster_whisper import WhisperModel


# =============================================================================
# 定数
# =============================================================================

class Config:
    """アプリケーション設定"""
    # Whisper設定
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"
    WHISPER_BEAM_SIZE = 5

    # 翻訳設定
    CLAUDE_MODEL = "claude-sonnet-4-20250514"
    TRANSLATION_MAX_TOKENS = 1024
    TRANSLATION_RETRY_DELAY = 2.0  # 秒
    TRANSLATION_REQUEST_INTERVAL = 0.1  # 秒

    # 音声抽出設定
    AUDIO_CODEC = "pcm_s16le"
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1


class Language:
    """言語設定"""
    AUTO = None
    JAPANESE = "ja"
    ENGLISH = "en"

    DISPLAY_MAP = {
        "自動検出": AUTO,
        "日本語": JAPANESE,
        "英語": ENGLISH,
    }


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class TranscriptionSegment:
    """文字起こしセグメント"""
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """文字起こし結果"""
    segments: list[TranscriptionSegment]
    detected_language: Optional[str]
    language_probability: Optional[float]


# =============================================================================
# 音声処理
# =============================================================================

def extract_audio(video_path: str, audio_path: str) -> None:
    """動画から音声を抽出"""
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(
        stream.audio,
        audio_path,
        acodec=Config.AUDIO_CODEC,
        ar=Config.AUDIO_SAMPLE_RATE,
        ac=Config.AUDIO_CHANNELS,
    )
    ffmpeg.run(stream, overwrite_output=True, quiet=True)


# =============================================================================
# 文字起こし
# =============================================================================

def load_whisper_model(model_name: str) -> WhisperModel:
    """Whisperモデルをロード"""
    return WhisperModel(
        model_name,
        device=Config.WHISPER_DEVICE,
        compute_type=Config.WHISPER_COMPUTE_TYPE,
    )


def transcribe_audio(
    model: WhisperModel,
    audio_path: str,
    language: Optional[str] = None,
) -> TranscriptionResult:
    """音声を文字起こし"""
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=Config.WHISPER_BEAM_SIZE,
    )

    segment_list = [
        TranscriptionSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
        )
        for seg in segments
    ]

    return TranscriptionResult(
        segments=segment_list,
        detected_language=info.language if language is None else None,
        language_probability=info.language_probability if language is None else None,
    )


# =============================================================================
# 翻訳
# =============================================================================

ProgressCallback = Callable[[int, int, str], None]


def _call_claude_api(client: anthropic.Anthropic, text: str) -> str:
    """Claude APIを呼び出して翻訳"""
    message = client.messages.create(
        model=Config.CLAUDE_MODEL,
        max_tokens=Config.TRANSLATION_MAX_TOKENS,
        system="あなたはプロの翻訳者です。英語を自然な日本語に翻訳してください。字幕用なので簡潔に。翻訳結果のみを出力し、説明は不要です。",
        messages=[
            {"role": "user", "content": f"次の英語を日本語に翻訳してください:\n{text}"}
        ],
    )
    return message.content[0].text.strip()


def _translate_single_segment(client: anthropic.Anthropic, text: str) -> str:
    """単一セグメントを翻訳（リトライ付き）"""
    try:
        return _call_claude_api(client, text)
    except anthropic.RateLimitError:
        time.sleep(Config.TRANSLATION_RETRY_DELAY)
        try:
            return _call_claude_api(client, text)
        except Exception as e:
            return f"[翻訳エラー: {str(e)[:20]}]"
    except Exception as e:
        return f"[翻訳エラー: {str(e)[:20]}]"


def translate_segments(
    segments: list[TranscriptionSegment],
    api_key: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[str]:
    """セグメントリストを翻訳"""
    client = anthropic.Anthropic(api_key=api_key)
    translated_texts = []
    total = len(segments)

    for i, segment in enumerate(segments):
        if progress_callback:
            preview = segment.text[:30] + "..." if len(segment.text) > 30 else segment.text
            progress_callback(i + 1, total, preview)

        translated = _translate_single_segment(client, segment.text)
        translated_texts.append(translated)

        time.sleep(Config.TRANSLATION_REQUEST_INTERVAL)

    return translated_texts


# =============================================================================
# SRT生成
# =============================================================================

def _format_srt_timestamp(seconds: float) -> str:
    """秒数をSRT形式のタイムスタンプに変換 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp(seconds: float) -> str:
    """秒数を HH:MM:SS 形式に変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _split_text_by_length(text: str, max_chars: int) -> list[str]:
    """テキストを最大文字数で分割（単語の途中で切らない）"""
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {word}" if current_chunk else word
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text]


def create_srt(segments: list[TranscriptionSegment], max_chars: int = 100) -> str:
    """SRT形式の字幕を生成（文字数制限付き）"""
    lines = []
    idx = 1

    for segment in segments:
        chunks = _split_text_by_length(segment.text, max_chars)
        chunk_duration = segment.duration / len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_start = segment.start + (i * chunk_duration)
            chunk_end = segment.start + ((i + 1) * chunk_duration)

            lines.append(str(idx))
            lines.append(f"{_format_srt_timestamp(chunk_start)} --> {_format_srt_timestamp(chunk_end)}")
            lines.append(chunk)
            lines.append("")
            idx += 1

    return "\n".join(lines)


def create_bilingual_srt(
    segments: list[TranscriptionSegment],
    translated_texts: list[str],
) -> str:
    """英語（上）+ 日本語（下）のバイリンガルSRTを生成"""
    lines = []

    for idx, (segment, ja_text) in enumerate(zip(segments, translated_texts), 1):
        lines.append(str(idx))
        lines.append(f"{_format_srt_timestamp(segment.start)} --> {_format_srt_timestamp(segment.end)}")
        lines.append(segment.text)
        lines.append(ja_text)
        lines.append("")

    return "\n".join(lines)


def create_timestamped_text(segments: list[TranscriptionSegment]) -> str:
    """タイムスタンプ付きテキストを生成"""
    lines = [
        f"[{_format_timestamp(seg.start)} - {_format_timestamp(seg.end)}] {seg.text}"
        for seg in segments
    ]
    return "\n".join(lines)


def create_full_text(segments: list[TranscriptionSegment]) -> str:
    """全文テキストを生成"""
    return "\n".join(seg.text for seg in segments)


# =============================================================================
# ユーティリティ
# =============================================================================

def get_api_key(input_key: Optional[str] = None) -> Optional[str]:
    """APIキーを取得（入力値 or 環境変数）"""
    return input_key or os.environ.get("ANTHROPIC_API_KEY")
