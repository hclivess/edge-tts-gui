"""
Synthesis core for edge-tts-gui: text chunking, per-chunk synthesis with retries, subtitle assembly
across chunks, audio concatenation and optional conversion with FFmpeg.
"""
import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import edge_tts

from config import MAX_CHUNK_CHARS
from utils import childproc


def part_name(path: str) -> str:
    """Temporary name next to the destination: out.mp3 -> out.part.mp3. Never leave a partial file under the final name."""
    base, ext = os.path.splitext(path)
    return f"{base}.part{ext}"


def write_atomic(path: str, data, mode: str = "w", encoding: Optional[str] = "utf-8") -> str:
    """Write to <name>.part.<ext> and os.replace into place once complete."""
    tmp = part_name(path)
    try:
        with open(tmp, mode, encoding=encoding if "b" not in mode else None) as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        _remove(tmp)
        raise
    return path


def _remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.abspath(__file__))


def find_ffmpeg() -> Optional[str]:
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    local = os.path.join(app_dir(), exe)
    if os.path.isfile(local):
        return local
    return shutil.which("ffmpeg")


# --------------------------------------------------------------------------- text
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+")


def split_text(text: str, method: str = "sentences", max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split into chunks <= max_chars at sentence / paragraph / line boundaries (falls back to harder splits)"""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    if method == "paragraphs":
        units = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        joiner = "\n\n"
    elif method == "lines":
        units = [l.strip() for l in text.split("\n") if l.strip()]
        joiner = "\n"
    else:
        units = [s.strip() for s in _SENTENCE_SPLIT.split(re.sub(r"\s*\n\s*", " ", text)) if s.strip()]
        joiner = " "
    chunks: List[str] = []
    cur = ""
    for u in units:
        if len(u) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            sub = split_text(u, "sentences", max_chars) if method != "sentences" else _hard_split(u, max_chars)
            chunks.extend(sub)
            continue
        cand = f"{cur}{joiner}{u}" if cur else u
        if len(cand) > max_chars:
            chunks.append(cur)
            cur = u
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks


def _hard_split(text: str, max_chars: int) -> List[str]:
    out, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}" if cur else word
        if len(cand) > max_chars and cur:
            out.append(cur)
            cur = word
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def strip_html(text: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>|</li>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    import html
    return html.unescape(re.sub(r"[ \t]+", " ", text))


def read_text_file(path: str) -> str:
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "cp1250", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if path.lower().endswith((".html", ".htm")):
        text = strip_html(text)
    elif path.lower().endswith(".srt"):
        text = "\n".join(l for l in text.splitlines()
                         if l.strip() and not l.strip().isdigit() and "-->" not in l)
    return text


# --------------------------------------------------------------------------- mp3 duration
_BITRATES = {  # (version_index, layer_index) -> table ; only Layer III used here
    (3, 1): [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],   # MPEG1 L3
    (2, 1): [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],        # MPEG2/2.5 L3
}
_SAMPLE_RATES = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def mp3_duration_ms(data: bytes) -> float:
    """Walk the MPEG frames and sum their durations (handles the CBR mono mp3 edge-tts produces)"""
    i, n, total = 0, len(data), 0.0
    if data[:3] == b"ID3":
        size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
        i = 10 + size
    while i + 4 <= n:
        if data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
            i += 1
            continue
        ver = (data[i + 1] >> 3) & 3          # 3=MPEG1, 2=MPEG2, 0=MPEG2.5
        layer = (data[i + 1] >> 1) & 3        # 1 = Layer III
        br_idx = data[i + 2] >> 4
        sr_idx = (data[i + 2] >> 2) & 3
        pad = (data[i + 2] >> 1) & 1
        if ver == 1 or layer != 1 or br_idx in (0, 15) or sr_idx == 3:
            i += 1
            continue
        table = _BITRATES[(3, 1)] if ver == 3 else _BITRATES[(2, 1)]
        bitrate = table[br_idx] * 1000
        sr = _SAMPLE_RATES[ver][sr_idx]
        spf = 1152 if ver == 3 else 576
        frame_len = int(spf / 8 * bitrate / sr) + pad
        if frame_len <= 0:
            i += 1
            continue
        total += spf / sr * 1000.0
        i += frame_len
    return total


# --------------------------------------------------------------------------- subtitles
@dataclass
class Cue:
    start_ms: float
    end_ms: float
    text: str


def _ts_srt(ms: float) -> str:
    ms = max(0, int(round(ms)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cues_to_srt(cues: List[Cue]) -> str:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{_ts_srt(c.start_ms)} --> {_ts_srt(c.end_ms)}\n{c.text.strip()}\n")
    return "\n".join(out)


def cues_to_vtt(cues: List[Cue]) -> str:
    lines = []
    for c_i, c in enumerate(cues, 1):
        lines.append(f"{c_i}\n{_ts_srt(c.start_ms).replace(',', '.')} --> {_ts_srt(c.end_ms).replace(',', '.')}\n{c.text.strip()}\n")
    return "WEBVTT\n\n" + "\n".join(lines)


# --------------------------------------------------------------------------- synthesis
@dataclass
class SynthResult:
    audio_path: str
    subtitle_paths: List[str] = field(default_factory=list)
    chunks: int = 0
    duration_ms: float = 0.0


class Synthesizer:
    def __init__(self, settings: Dict, log: Callable[[str], None] = lambda m: None,
                 progress: Callable[[int, int], None] = lambda d, t: None,
                 should_stop: Callable[[], bool] = lambda: False):
        self.s = settings
        self.log = log
        self.progress = progress
        self.should_stop = should_stop
        self._proc = None

    def stop(self):
        """Kill a running ffmpeg conversion (the caller sets its stop flag first; temp outputs are removed)."""
        proc = self._proc
        if proc is not None:
            childproc.kill(proc)

    # -- one chunk --------------------------------------------------------------
    async def _synth_chunk(self, text: str) -> (bytes, List[Cue]):
        kw = dict(rate=f"{int(self.s['rate']):+d}%", volume=f"{int(self.s['volume']):+d}%",
                  pitch=f"{int(self.s['pitch']):+d}Hz", boundary=self.s.get("boundary", "SentenceBoundary"))
        if self.s.get("proxy"):
            kw["proxy"] = self.s["proxy"]
        comm = edge_tts.Communicate(text, self.s["voice"], **kw)
        audio = bytearray()
        cues: List[Cue] = []
        async for ch in comm.stream():
            if self.should_stop():
                raise InterruptedError("stopped")
            if ch["type"] == "audio":
                audio += ch["data"]
            elif ch["type"] in ("WordBoundary", "SentenceBoundary"):
                start = ch["offset"] / 10000.0
                cues.append(Cue(start, start + ch["duration"] / 10000.0, ch["text"]))
        if not audio:
            raise RuntimeError("no audio received")
        return bytes(audio), cues

    def _synth_chunk_retry(self, text: str, label: str) -> (bytes, List[Cue]):
        retries = max(1, int(self.s.get("max_retries", 3)))
        delay = float(self.s.get("retry_delay", 2))
        last = None
        for attempt in range(1, retries + 1):
            if self.should_stop():
                raise InterruptedError("stopped")
            try:
                return asyncio.run(self._synth_chunk(text))
            except InterruptedError:
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                self.log(f"{label}: attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries:
                    time.sleep(delay)
        raise RuntimeError(f"{label}: failed after {retries} attempts: {last}")

    # -- whole text ---------------------------------------------------------------
    def synthesize(self, text: str, out_base: str) -> SynthResult:
        """out_base: output path without extension"""
        chunks = split_text(text, self.s.get("split_method", "sentences"), int(self.s.get("chunk_chars", MAX_CHUNK_CHARS)))
        if not chunks:
            raise ValueError("empty text")
        if len(chunks) > 1:
            self.log(f"{len(chunks)} chunks of up to {self.s.get('chunk_chars', MAX_CHUNK_CHARS)} characters")
        pause_ms = int(self.s.get("pause_between_chunks_ms", 0) or 0)
        fmt = self.s.get("format", "mp3")
        mp3_path = out_base + ".mp3"
        mp3_tmp = part_name(mp3_path)
        cues: List[Cue] = []
        offset = 0.0
        # audio chunks are appended to <name>.part.mp3 as they arrive; the file only gets its final name when complete
        try:
            with open(mp3_tmp, "wb") as fh:
                for i, chunk in enumerate(chunks, 1):
                    self.progress(i - 1, len(chunks))
                    data, chunk_cues = self._synth_chunk_retry(chunk, f"chunk {i}/{len(chunks)}")
                    fh.write(data)
                    for c in chunk_cues:
                        cues.append(Cue(c.start_ms + offset, c.end_ms + offset, c.text))
                    offset += mp3_duration_ms(data)
                    if pause_ms and i < len(chunks):
                        fh.write(self._silence_mp3(pause_ms))
                        offset += pause_ms
                    self.progress(i, len(chunks))
            if self.should_stop():
                raise InterruptedError("stopped")
            os.replace(mp3_tmp, mp3_path)
        except BaseException:
            _remove(mp3_tmp)
            raise

        audio_path = mp3_path
        if fmt != "mp3":
            audio_path = self._convert(mp3_path, out_base + "." + fmt)
            if audio_path != mp3_path:
                os.remove(mp3_path)
        subs = []
        mode = self.s.get("subtitles", "none")
        if cues and mode in ("srt", "both"):
            subs.append(write_atomic(out_base + ".srt", cues_to_srt(cues)))
        if cues and mode in ("vtt", "both"):
            subs.append(write_atomic(out_base + ".vtt", cues_to_vtt(cues)))
        return SynthResult(audio_path, subs, len(chunks), offset)

    _silence_cache: Dict[int, bytes] = {}

    def _silence_mp3(self, ms: int) -> bytes:
        """Silence as mp3 frames matching edge-tts output (24 kHz mono 48 kbps) via ffmpeg; empty if unavailable"""
        if ms in self._silence_cache:
            return self._silence_cache[ms]
        ff = find_ffmpeg()
        if not ff:
            self.log("ffmpeg not found - pause between chunks skipped")
            self._silence_cache[ms] = b""
            return b""
        res = childproc.run([ff, "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", f"{ms / 1000:.3f}",
                             "-c:a", "libmp3lame", "-b:a", "48k", "-f", "mp3", "-"])
        data = res.stdout if res.returncode == 0 else b""
        self._silence_cache[ms] = data
        return data

    def _convert(self, src: str, dst: str) -> str:
        ff = find_ffmpeg()
        if not ff:
            self.log(f"ffmpeg not found - keeping mp3 instead of {os.path.splitext(dst)[1]}")
            return src
        tmp = part_name(dst)
        proc = childproc.popen([ff, "-y", "-v", "error", "-i", src, tmp], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
        self._proc = proc
        try:
            _, err = proc.communicate()
        finally:
            self._proc = None
            childproc.forget(proc)
        if self.should_stop():
            _remove(tmp)
            raise InterruptedError("stopped")
        if proc.returncode != 0:
            _remove(tmp)
            self.log(f"conversion failed ({(err or '').strip()[:200]}) - keeping mp3")
            return src
        os.replace(tmp, dst)
        return dst


# --------------------------------------------------------------------------- voices
def list_voices() -> List[Dict]:
    return asyncio.run(edge_tts.list_voices())


def voice_display(v: Dict) -> str:
    return f"{v.get('ShortName', '?')}  —  {v.get('LocaleName', v.get('Locale', ''))}, {v.get('Gender', '')}"
