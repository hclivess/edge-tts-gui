"""
edge-tts-gui - configuration and constants
"""

APP_NAME = "edge-tts-gui"
APP_VERSION = "2.0.2"
WINDOW_MIN_WIDTH = 900
WINDOW_MIN_HEIGHT = 600

TEXT_EXTENSIONS = [".txt", ".md", ".text", ".rst", ".log", ".srt", ".html", ".htm"]

AUDIO_FORMATS = ["mp3", "wav", "ogg", "flac", "m4a"]        # mp3 native; others need ffmpeg
SUBTITLE_FORMATS = ["none", "srt", "vtt", "both"]
BOUNDARIES = {"SentenceBoundary": "sentence cues", "WordBoundary": "word cues"}
SPLIT_METHODS = {"sentences": "Sentences", "paragraphs": "Paragraphs", "lines": "Lines"}

DEFAULT_VOICE = "en-US-AriaNeural"
MAX_CHUNK_CHARS = 3000          # edge-tts rejects very long single requests

DEFAULT_SETTINGS = {
    "voice": DEFAULT_VOICE,
    "language_filter": "all",
    "rate": 0,                  # percent
    "volume": 0,                # percent
    "pitch": 0,                 # Hz
    "split_method": "sentences",
    "chunk_chars": MAX_CHUNK_CHARS,
    "pause_between_chunks_ms": 0,
    "max_retries": 3,
    "retry_delay": 2,
    "format": "mp3",
    "subtitles": "srt",
    "boundary": "SentenceBoundary",
    "output_mode": "same",      # same | custom
    "output_dir": "",
    "suffix": "",
    "overwrite": False,
    "proxy": "",
}
