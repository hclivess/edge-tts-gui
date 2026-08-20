#!/usr/bin/env python3
"""
edge-tts-gui - batch text-to-speech with Microsoft Edge neural voices (PySide6)

Queue of text files (or pasted text) -> mp3 / wav / ogg / flac / m4a + srt / vtt subtitles.
Set EDGE_TTS_GUI_SELFTEST=<text file> for a headless smoke test (synthesize it, exit 0 on success).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QIcon, QTextCursor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from config import (APP_NAME, APP_VERSION, AUDIO_FORMATS, BOUNDARIES, DEFAULT_SETTINGS, DEFAULT_VOICE,  # noqa: E402
                    SPLIT_METHODS, SUBTITLE_FORMATS, TEXT_EXTENSIONS, WINDOW_MIN_HEIGHT, WINDOW_MIN_WIDTH)
from tts_engine import Synthesizer, app_dir, find_ffmpeg, list_voices, read_text_file, split_text, voice_display  # noqa: E402

_BTN = """
    QPushButton { font-size: 14px; font-weight: bold; padding: 8px; background-color: %s; color: %s; border-radius: 5px; }
    QPushButton:hover { background-color: %s; }
    QPushButton:disabled { background-color: #cccccc; color: #666; }
"""
_BAR = """
    QProgressBar { border: 1px solid #c8c8c8; border-radius: 4px; background: #f0f0f0;
                   text-align: center; height: 18px; font-weight: bold; color: #333; }
    QProgressBar::chunk { background-color: %s; border-radius: 3px; }
"""
_STATUS_COLORS = {"pending": "#ffffff", "running": "#fff3cd", "done": "#d4edda", "failed": "#f8d7da",
                  "stopped": "#e2e3e5"}
PRESETS = {
    "audiobook": {"name": "Audiobook (sentence subtitles, mp3)",
                  "settings": {"format": "mp3", "subtitles": "srt", "boundary": "SentenceBoundary",
                               "split_method": "paragraphs", "pause_between_chunks_ms": 400, "rate": 0}},
    "video": {"name": "Video voice-over (wav + word-level vtt)",
              "settings": {"format": "wav", "subtitles": "vtt", "boundary": "WordBoundary",
                           "split_method": "sentences", "pause_between_chunks_ms": 0}},
    "fast": {"name": "Quick listen (mp3, +20 % rate, no subtitles)",
             "settings": {"format": "mp3", "subtitles": "none", "rate": 20, "pause_between_chunks_ms": 0}},
}


class QueueItem:
    def __init__(self, path: str = "", text: str = "", title: str = ""):
        self.path = path
        self.text = text
        self.title = title or (os.path.basename(path) if path else "pasted text")

    def read(self) -> str:
        return read_text_file(self.path) if self.path else self.text

    def out_base(self, settings: Dict) -> str:
        stem = Path(self.path).stem if self.path else self.title.replace(" ", "_")
        folder = os.path.dirname(os.path.abspath(self.path)) if self.path else os.getcwd()
        if settings["output_mode"] == "custom" and settings["output_dir"]:
            folder = settings["output_dir"]
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, stem + settings.get("suffix", ""))


class Worker(QThread):
    log = Signal(str)
    status = Signal(str)
    item_started = Signal(int)
    item_progress = Signal(int, int)
    item_done = Signal(int, str, str)      # index, state, detail
    finished_all = Signal(int, int)

    def __init__(self, items: List[QueueItem], settings: Dict):
        super().__init__()
        self.items = items
        self.settings = settings
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        ok = 0
        synth = Synthesizer(self.settings, log=self.log.emit, progress=self.item_progress.emit,
                            should_stop=lambda: self._stop)
        for i, item in enumerate(self.items):
            if self._stop:
                self.item_done.emit(i, "stopped", "")
                continue
            self.item_started.emit(i)
            try:
                text = item.read().strip()
                if not text:
                    raise ValueError("file is empty")
                base = item.out_base(self.settings)
                target = base + "." + self.settings["format"]
                if os.path.exists(target) and not self.settings.get("overwrite"):
                    self.item_done.emit(i, "done", f"exists, skipped: {os.path.basename(target)}")
                    ok += 1
                    continue
                self.status.emit(f"Synthesizing {item.title}")
                self.log.emit(f"=== {item.title} ({len(text):,} characters)")
                res = synth.synthesize(text, base)
                ok += 1
                subs = f" + {', '.join(os.path.basename(p) for p in res.subtitle_paths)}" if res.subtitle_paths else ""
                self.log.emit(f"{os.path.basename(res.audio_path)}{subs} — {res.duration_ms / 60000:.1f} min, "
                              f"{res.chunks} chunk(s)")
                self.item_done.emit(i, "done", f"{res.duration_ms / 60000:.1f} min")
            except InterruptedError:
                self.item_done.emit(i, "stopped", "")
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"ERROR {item.title}: {exc}")
                self.log.emit(traceback.format_exc(limit=2))
                self.item_done.emit(i, "failed", str(exc)[:80])
        self.finished_all.emit(ok, len(self.items))


class VoiceLoader(QThread):
    done = Signal(list, str)

    def run(self):
        last = ""
        for attempt in range(3):
            try:
                self.done.emit(list_voices(), "")
                return
            except Exception as exc:  # noqa: BLE001
                last = str(exc)
                time.sleep(2)
        self.done.emit([], last)


class PreviewThread(QThread):
    done = Signal(str, str)

    def __init__(self, settings: Dict, text: str):
        super().__init__()
        self.settings, self.text = settings, text

    def run(self):
        try:
            base = os.path.join(tempfile.gettempdir(), "edge-tts-gui-preview")
            s = dict(self.settings, format="mp3", subtitles="none")
            res = Synthesizer(s).synthesize(self.text, base)
            self.done.emit(res.audio_path, "")
        except Exception as exc:  # noqa: BLE001
            self.done.emit("", str(exc))


class QueueTable(QTableWidget):
    dropped = Signal(list, str)     # urls, text

    def __init__(self):
        super().__init__(0, 3)
        self.setHorizontalHeaderLabels(["Text", "Status", "Result"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAcceptDrops(True)
        self.setStyleSheet("QTableWidget { border: 2px solid #aaa; border-radius: 5px; background: #f9f9f9; }")

    def dragEnterEvent(self, e: QDragEnterEvent):
        e.accept() if (e.mimeData().hasUrls() or e.mimeData().hasText()) else e.ignore()

    def dragMoveEvent(self, e):
        e.accept() if (e.mimeData().hasUrls() or e.mimeData().hasText()) else e.ignore()

    def dropEvent(self, e: QDropEvent):
        m = e.mimeData()
        e.accept()
        if m.hasUrls():
            self.dropped.emit([u.toLocalFile() for u in m.urls()], "")
        elif m.hasText():
            self.dropped.emit([], m.text())


class TextDialog(QDialog):
    def __init__(self, parent, text: str = "", title: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Add text")
        self.resize(600, 400)
        lay = QVBoxLayout(self)
        self.title_edit = QLineEdit(title)
        self.title_edit.setPlaceholderText("name for the output file (without extension)")
        lay.addWidget(QLabel("Name:"))
        lay.addWidget(self.title_edit)
        self.edit = QPlainTextEdit(text)
        self.edit.setPlaceholderText("Paste or type the text to speak…")
        lay.addWidget(QLabel("Text:"))
        lay.addWidget(self.edit, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(1180, 760)
        self.qsettings = QSettings(APP_NAME, "Settings")
        self.presets_dir = os.path.join(app_dir(), "presets")
        os.makedirs(self.presets_dir, exist_ok=True)
        self.cache_file = os.path.join(self.presets_dir, "voices_cache.json")
        self.items: List[QueueItem] = []
        self.voices: List[Dict] = []
        self.worker: Optional[Worker] = None
        self.controls: Dict[str, QWidget] = {}
        self._run_start = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self._build_menu()
        self.apply_settings(dict(DEFAULT_SETTINGS))
        self._load_persisted()
        self._load_voices()
        if not find_ffmpeg():
            self._log("ffmpeg not found: output is limited to mp3 and pauses between chunks are skipped. "
                      "Install FFmpeg (or put ffmpeg next to the app) for wav / ogg / flac / m4a.")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._left_panel())
        split.addWidget(self._right_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([640, 520])
        root.addWidget(split, 1)
        self.statusBar().showMessage("Ready")

    def _left_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("<b>Queue</b> — drop text files, folders or selected text here"))
        self.table = QueueTable()
        self.table.dropped.connect(self._on_drop)
        self.table.doubleClicked.connect(lambda idx: self._edit_item(idx.row()))
        lay.addWidget(self.table, 3)
        row = QHBoxLayout()
        for text, fn in (("Add files", self.add_files), ("Add folder", self.add_folder), ("Add text…", self.add_text),
                         ("Remove", self.remove_selected), ("Clear", self.clear_queue)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        open_out = QPushButton("Open output folder")
        open_out.clicked.connect(self.open_output)
        row.addWidget(open_out)
        lay.addLayout(row)

        prog = QGroupBox("Progress")
        pl = QVBoxLayout(prog)
        top = QHBoxLayout()
        self.phase_label = QLabel("Idle")
        self.phase_label.setStyleSheet("font-weight: bold;")
        self.counter_label = QLabel("")
        self.counter_label.setStyleSheet("color: #666;")
        top.addWidget(self.phase_label)
        top.addStretch()
        top.addWidget(self.counter_label)
        pl.addLayout(top)
        self.file_bar = QProgressBar()
        self.file_bar.setStyleSheet(_BAR % "#4a90d9")
        self.file_bar.setFormat("%v / %m chunks")
        pl.addWidget(self.file_bar)
        self.total_bar = QProgressBar()
        self.total_bar.setStyleSheet(_BAR % "#28a745")
        self.total_bar.setFormat("%v / %m items")
        pl.addWidget(self.total_bar)
        stats = QHBoxLayout()
        self.stat_done = self._stat(stats, "done")
        self.stat_failed = self._stat(stats, "failed")
        self.stat_elapsed = self._stat(stats, "elapsed")
        self.stat_voice = self._stat(stats, "voice")
        pl.addLayout(stats)
        lay.addWidget(prog)

        lay.addWidget(QLabel("<b>Log</b>"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("QTextEdit { font-family: monospace; font-size: 12px; background: #fcfcfc; }")
        lay.addWidget(self.log, 2)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setStyleSheet(_BTN % ("#28a745", "white", "#218838"))
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setStyleSheet(_BTN % ("#dc3545", "white", "#c82333"))
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        btns.addWidget(self.start_btn, 2)
        btns.addWidget(self.stop_btn, 1)
        lay.addLayout(btns)
        return w

    @staticmethod
    def _stat(layout, caption) -> QLabel:
        box = QVBoxLayout()
        c = QLabel(caption)
        c.setStyleSheet("color: #888; font-size: 10px;")
        v = QLabel("--")
        v.setStyleSheet("font-weight: bold; font-size: 13px;")
        box.addWidget(c)
        box.addWidget(v)
        layout.addLayout(box)
        return v

    @staticmethod
    def _hint(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet("color: #666; font-size: 11px;")
        return lab

    @staticmethod
    def _wrap(widget: QWidget) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(widget)
        sa.setFrameShape(QScrollArea.Shape.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return sa

    def _right_panel(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._wrap(self._voice_tab()), "Voice")
        self.tabs.addTab(self._wrap(self._text_tab()), "Text")
        self.tabs.addTab(self._wrap(self._output_tab()), "Output")
        for w in self.controls.values():
            if isinstance(w, QComboBox):
                w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                w.setMinimumContentsLength(8)
            if isinstance(w, (QComboBox, QLineEdit, QSpinBox)) and w.maximumWidth() > 1000:
                w.setMinimumWidth(0)
                w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        return self.tabs

    def _slider_row(self, key: str, lo: int, hi: int, unit: str) -> QHBoxLayout:
        row = QHBoxLayout()
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(lo, hi)
        sl.setTickInterval((hi - lo) // 10)
        val = QSpinBox()
        val.setRange(lo, hi)
        val.setSuffix(unit)
        val.setFixedWidth(80)
        sl.valueChanged.connect(val.setValue)
        val.valueChanged.connect(sl.setValue)
        row.addWidget(sl, 1)
        row.addWidget(val)
        self.controls[key] = val
        return row

    def _voice_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        c = self.controls
        g = QGroupBox("Voice")
        f = QFormLayout(g)
        c["language_filter"] = QComboBox()
        c["language_filter"].addItem("All languages", "all")
        c["language_filter"].currentIndexChanged.connect(self._populate_voices)
        f.addRow("Language:", c["language_filter"])
        c["voice"] = QComboBox()
        c["voice"].setEditable(True)
        c["voice"].currentIndexChanged.connect(lambda *_: self.stat_voice.setText(self._voice_short()))
        f.addRow("Voice:", c["voice"])
        row = QHBoxLayout()
        self.voice_label = self._hint("")
        self.voice_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        refresh = QPushButton("Refresh voice list")
        refresh.clicked.connect(lambda: self._load_voices(force=True))
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setToolTip("Speak a short sample with the current voice and settings")
        self.preview_btn.clicked.connect(self._preview)
        row.addWidget(self.voice_label, 1)
        row.addWidget(self.preview_btn)
        row.addWidget(refresh)
        f.addRow(row)
        lay.addWidget(g)

        g2 = QGroupBox("Prosody")
        f2 = QFormLayout(g2)
        f2.addRow("Rate:", self._slider_row("rate", -100, 100, " %"))
        f2.addRow("Volume:", self._slider_row("volume", -100, 100, " %"))
        f2.addRow("Pitch:", self._slider_row("pitch", -100, 100, " Hz"))
        reset = QPushButton("Reset prosody")
        reset.clicked.connect(lambda: [c[k].setValue(0) for k in ("rate", "volume", "pitch")])
        f2.addRow("", reset)
        lay.addWidget(g2)

        g3 = QGroupBox("Network")
        f3 = QFormLayout(g3)
        c["max_retries"] = QSpinBox()
        c["max_retries"].setRange(1, 10)
        f3.addRow("Retries per chunk:", c["max_retries"])
        c["retry_delay"] = QSpinBox()
        c["retry_delay"].setRange(0, 60)
        c["retry_delay"].setSuffix(" s")
        f3.addRow("Retry delay:", c["retry_delay"])
        c["proxy"] = QLineEdit()
        c["proxy"].setPlaceholderText("optional, e.g. http://127.0.0.1:8080")
        f3.addRow("Proxy:", c["proxy"])
        lay.addWidget(g3)
        lay.addStretch()
        return w

    def _text_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        c = self.controls
        g = QGroupBox("Long texts")
        f = QFormLayout(g)
        c["split_method"] = QComboBox()
        for k, v in SPLIT_METHODS.items():
            c["split_method"].addItem(v, k)
        f.addRow("Split at:", c["split_method"])
        c["chunk_chars"] = QSpinBox()
        c["chunk_chars"].setRange(200, 5000)
        c["chunk_chars"].setSingleStep(100)
        c["chunk_chars"].setSuffix(" chars")
        f.addRow("Max chunk:", c["chunk_chars"])
        c["pause_between_chunks_ms"] = QSpinBox()
        c["pause_between_chunks_ms"].setRange(0, 5000)
        c["pause_between_chunks_ms"].setSingleStep(100)
        c["pause_between_chunks_ms"].setSuffix(" ms")
        c["pause_between_chunks_ms"].setToolTip("Silence inserted between chunks (needs ffmpeg); useful with paragraph splitting")
        f.addRow("Pause between chunks:", c["pause_between_chunks_ms"])
        f.addRow(self._hint("The service accepts a few thousand characters per request. Longer texts are split at "
                            "sentence / paragraph / line boundaries, synthesized chunk by chunk (each with retries) and "
                            "joined into one file; subtitle timings are shifted so they stay in sync."))
        lay.addWidget(g)
        g2 = QGroupBox("Input files")
        f2 = QFormLayout(g2)
        f2.addRow(self._hint("Accepted: " + ", ".join(TEXT_EXTENSIONS) + ". HTML is stripped to plain text, .srt "
                             "loses its numbering and timestamps. Encoding is auto-detected (UTF-8, cp1250, cp1252)."))
        lay.addWidget(g2)
        lay.addStretch()
        return w

    def _output_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        c = self.controls
        g = QGroupBox("Audio")
        f = QFormLayout(g)
        c["format"] = QComboBox()
        c["format"].addItems(AUDIO_FORMATS)
        c["format"].setToolTip("mp3 is what the service sends; other formats are converted with ffmpeg")
        f.addRow("Format:", c["format"])
        c["subtitles"] = QComboBox()
        c["subtitles"].addItems(SUBTITLE_FORMATS)
        f.addRow("Subtitles:", c["subtitles"])
        c["boundary"] = QComboBox()
        for k, v in BOUNDARIES.items():
            c["boundary"].addItem(v, k)
        f.addRow("Subtitle cues:", c["boundary"])
        lay.addWidget(g)
        g2 = QGroupBox("Location")
        f2 = QFormLayout(g2)
        c["output_mode"] = QComboBox()
        c["output_mode"].addItem("Next to the text file", "same")
        c["output_mode"].addItem("Custom folder", "custom")
        f2.addRow("Save:", c["output_mode"])
        c["output_dir"] = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(c["output_dir"], 1)
        b = QPushButton("…")
        b.setFixedWidth(32)
        b.clicked.connect(lambda: self._browse_dir(c["output_dir"]))
        row.addWidget(b)
        f2.addRow("Folder:", row)
        c["suffix"] = QLineEdit()
        c["suffix"].setPlaceholderText("e.g. _aria")
        f2.addRow("File suffix:", c["suffix"])
        c["overwrite"] = QCheckBox("Overwrite existing files")
        f2.addRow("", c["overwrite"])
        lay.addWidget(g2)
        lay.addStretch()
        return w

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        fm.addAction("Add files…", self.add_files, "Ctrl+O")
        fm.addAction("Add folder…", self.add_folder, "Ctrl+Shift+O")
        fm.addAction("Add text…", self.add_text, "Ctrl+T")
        fm.addAction("Open output folder", self.open_output)
        fm.addSeparator()
        fm.addAction("Exit", self.close, "Ctrl+Q")
        self.presets_menu = mb.addMenu("&Presets")
        self._rebuild_presets_menu()
        mb.addMenu("&Help").addAction("About", self._about)

    def _rebuild_presets_menu(self):
        m = self.presets_menu
        m.clear()
        for key, p in PRESETS.items():
            m.addAction(p["name"], lambda k=key: self._apply_preset(k))
        m.addSeparator()
        user = sorted(f[:-5] for f in os.listdir(self.presets_dir)
                      if f.endswith(".json") and f not in ("defaults.json", "voices_cache.json"))
        for name in user:
            m.addAction(name, lambda n=name: self._load_preset_file(os.path.join(self.presets_dir, n + ".json")))
        if user:
            m.addSeparator()
        m.addAction("Save preset…", self._save_preset)
        m.addAction("Delete preset…", self._delete_preset)
        m.addAction("Import preset…", self._import_preset)
        m.addAction("Export preset…", self._export_preset)
        m.addSeparator()
        m.addAction("Save current settings as defaults", self._save_defaults)
        m.addAction("Reset to factory defaults", lambda: self.apply_settings(dict(DEFAULT_SETTINGS)))

    # ------------------------------------------------------------- settings
    def get_settings(self) -> Dict:
        s = {}
        for key, w in self.controls.items():
            if isinstance(w, QCheckBox):
                s[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                if key == "voice":
                    data = w.currentData()
                    s[key] = data if data and w.currentText() == w.itemText(w.currentIndex()) else w.currentText().strip()
                else:
                    data = w.currentData()
                    s[key] = data if data is not None else w.currentText().strip()
            elif isinstance(w, QSpinBox):
                s[key] = w.value()
            elif isinstance(w, QLineEdit):
                s[key] = w.text().strip()
        return s

    def apply_settings(self, s: Dict):
        for key, w in self.controls.items():
            if key not in s:
                continue
            v = s[key]
            try:
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(v))
                elif isinstance(w, QComboBox):
                    idx = w.findData(v)
                    if idx < 0:
                        idx = w.findText(str(v))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                    elif w.isEditable():
                        w.setCurrentText(str(v))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(v))
                elif isinstance(w, QLineEdit):
                    w.setText("" if v is None else str(v))
            except Exception:
                pass
        if "voice" in s:
            self._pending_voice = s["voice"]

    def _load_persisted(self):
        self._pending_voice = DEFAULT_VOICE
        d = os.path.join(self.presets_dir, "defaults.json")
        if os.path.isfile(d):
            try:
                self.apply_settings(json.load(open(d, encoding="utf-8")))
            except Exception:
                pass
        raw = self.qsettings.value("settings")
        if raw:
            try:
                self.apply_settings(json.loads(raw))
            except Exception:
                pass
        geo = self.qsettings.value("geometry")
        if geo:
            self.restoreGeometry(geo)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Quit", "Synthesis is running. Stop and quit?") != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.request_stop()
            self.worker.wait(5000)
        self.qsettings.setValue("settings", json.dumps(self.get_settings()))
        self.qsettings.setValue("geometry", self.saveGeometry())
        event.accept()

    def _write_json(self, path, data) -> bool:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Preset", f"Could not write {path}:\n{exc}")
            return False

    def _apply_preset(self, key):
        self.apply_settings(PRESETS[key]["settings"])
        self.statusBar().showMessage(f"Preset applied: {PRESETS[key]['name']}")

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if ok and name.strip() and self._write_json(os.path.join(self.presets_dir, name.strip() + ".json"), self.get_settings()):
            self._rebuild_presets_menu()

    def _delete_preset(self):
        user = sorted(f[:-5] for f in os.listdir(self.presets_dir)
                      if f.endswith(".json") and f not in ("defaults.json", "voices_cache.json"))
        if not user:
            QMessageBox.information(self, "Presets", "No saved presets.")
            return
        name, ok = QInputDialog.getItem(self, "Delete preset", "Preset:", user, 0, False)
        if ok and name:
            os.remove(os.path.join(self.presets_dir, name + ".json"))
            self._rebuild_presets_menu()

    def _load_preset_file(self, path):
        try:
            self.apply_settings(json.load(open(path, encoding="utf-8")))
            self._select_voice(self._pending_voice)
            self.statusBar().showMessage(f"Preset loaded: {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.warning(self, "Preset", f"Could not read preset:\n{exc}")

    def _import_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import preset", "", "JSON (*.json)")
        if path:
            self._load_preset_file(path)

    def _export_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export preset", "preset.json", "JSON (*.json)")
        if path:
            self._write_json(path, self.get_settings())

    def _save_defaults(self):
        if self._write_json(os.path.join(self.presets_dir, "defaults.json"), self.get_settings()):
            self.statusBar().showMessage("Current settings saved as defaults")

    def _browse_dir(self, line: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Output folder", line.text() or os.path.expanduser("~"))
        if d:
            line.setText(d)
            self.controls["output_mode"].setCurrentIndex(1)

    # ---------------------------------------------------------------- voices
    def _load_voices(self, force: bool = False):
        if not force and os.path.isfile(self.cache_file):
            try:
                self.voices = json.load(open(self.cache_file, encoding="utf-8"))
                self._voices_ready(f"{len(self.voices)} voices (cached list — Refresh to update)")
                return
            except Exception:
                pass
        self.voice_label.setText("Loading voice list…")
        self._loader = VoiceLoader()

        def done(voices, err):
            if voices:
                self.voices = voices
                self._write_json(self.cache_file, voices)
                self._voices_ready(f"{len(voices)} voices")
            else:
                self.voice_label.setText(f"Could not load voices: {err}")
                self.controls["voice"].setEditable(True)
                self.controls["voice"].setCurrentText(self._pending_voice)
        self._loader.done.connect(done)
        self._loader.start()

    def _voices_ready(self, msg: str):
        lf = self.controls["language_filter"]
        cur = lf.currentData() or "all"
        lf.blockSignals(True)
        lf.clear()
        lf.addItem("All languages", "all")
        langs = {}
        for v in self.voices:
            loc = v.get("Locale", "")
            name = v.get("LocaleName", loc)
            langs.setdefault(loc.split("-")[0], name.split("(")[0].strip())
        for code, name in sorted(langs.items(), key=lambda kv: kv[1]):
            lf.addItem(f"{name} ({code})", code)
        idx = lf.findData(cur)
        lf.setCurrentIndex(idx if idx >= 0 else 0)
        lf.blockSignals(False)
        self._populate_voices()
        self._select_voice(self._pending_voice)
        self.voice_label.setText(msg)

    def _populate_voices(self, *_):
        vc = self.controls["voice"]
        lang = self.controls["language_filter"].currentData() or "all"
        current = vc.currentData() or vc.currentText()
        vc.blockSignals(True)
        vc.clear()
        for v in sorted(self.voices, key=lambda v: v.get("ShortName", "")):
            if lang == "all" or v.get("Locale", "").startswith(lang + "-"):
                vc.addItem(voice_display(v), v.get("ShortName"))
        vc.blockSignals(False)
        self._select_voice(current)

    def _select_voice(self, short_name: str):
        vc = self.controls["voice"]
        idx = vc.findData(short_name)
        if idx >= 0:
            vc.setCurrentIndex(idx)
        elif vc.count():
            # voice from another language filter: switch the filter to it
            match = next((v for v in self.voices if v.get("ShortName") == short_name), None)
            if match and self.controls["language_filter"].currentData() != "all":
                lf = self.controls["language_filter"]
                lf.setCurrentIndex(lf.findData(match["Locale"].split("-")[0]))
                return
            vc.setCurrentIndex(0)
        else:
            vc.setCurrentText(short_name)
        self.stat_voice.setText(self._voice_short())

    def _voice_short(self) -> str:
        vc = self.controls["voice"]
        return (vc.currentData() or vc.currentText()).replace("Neural", "")

    def _preview(self):
        s = self.get_settings()
        if not s["voice"]:
            return
        self.preview_btn.setEnabled(False)
        self.voice_label.setText("Synthesizing preview…")
        self._preview_thread = PreviewThread(s, "Hello! This is a short preview of the selected voice, rate and pitch.")

        def done(path, err):
            self.preview_btn.setEnabled(True)
            if err:
                self.voice_label.setText(f"Preview failed: {err}")
                return
            self.voice_label.setText("Preview played with the system player")
            self._open_path(path)
        self._preview_thread.done.connect(done)
        self._preview_thread.start()

    # ------------------------------------------------------------------ queue
    def _on_drop(self, urls: List[str], text: str):
        if urls:
            self.add_paths(urls)
        elif text.strip():
            self._add_item(QueueItem(text=text, title=f"dropped text {len(self.items) + 1}"))

    def add_paths(self, paths: List[str]):
        n = 0
        for p in paths:
            path = Path(p)
            if path.is_dir():
                for f in sorted(path.rglob("*")):
                    # folder scans skip .srt so our own subtitle outputs are never re-queued
                    if f.suffix.lower() in TEXT_EXTENSIONS and f.suffix.lower() != ".srt":
                        n += self._add_item(QueueItem(path=str(f)))
            elif path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
                n += self._add_item(QueueItem(path=str(path)))
        if n:
            self.statusBar().showMessage(f"{n} item(s) added — {len(self.items)} in queue")

    def _add_item(self, item: QueueItem) -> int:
        if item.path and any(i.path == item.path for i in self.items):
            return 0
        self.items.append(item)
        r = self.table.rowCount()
        self.table.insertRow(r)
        it = QTableWidgetItem(item.title)
        it.setToolTip(item.path or item.text[:500])
        self.table.setItem(r, 0, it)
        self.table.setItem(r, 1, QTableWidgetItem("pending"))
        self.table.setItem(r, 2, QTableWidgetItem(""))
        self._color_row(r, "pending")
        return 1

    def _edit_item(self, row: int):
        if self.worker and self.worker.isRunning():
            return
        item = self.items[row]
        if item.path:
            self._open_path(item.path)
            return
        dlg = TextDialog(self, item.text, item.title)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            item.text = dlg.edit.toPlainText()
            item.title = dlg.title_edit.text().strip() or item.title
            self.table.item(row, 0).setText(item.title)

    def _color_row(self, row: int, state: str):
        for col in range(3):
            it = self.table.item(row, col)
            if it:
                it.setBackground(QColor(_STATUS_COLORS.get(state, "#ffffff")))

    def add_files(self):
        exts = " ".join(f"*{e}" for e in TEXT_EXTENSIONS)
        files, _ = QFileDialog.getOpenFileNames(self, "Add text files", "", f"Text files ({exts});;All files (*)")
        self.add_paths(files)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder")
        if d:
            self.add_paths([d])

    def add_text(self):
        dlg = TextDialog(self, "", f"text {len(self.items) + 1}")
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.edit.toPlainText().strip():
            self._add_item(QueueItem(text=dlg.edit.toPlainText(), title=dlg.title_edit.text().strip() or "text"))

    def remove_selected(self):
        if self.worker and self.worker.isRunning():
            return
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
            del self.items[r]

    def clear_queue(self):
        if self.worker and self.worker.isRunning():
            return
        self.items.clear()
        self.table.setRowCount(0)

    def open_output(self):
        s = self.get_settings()
        if s["output_mode"] == "custom" and s["output_dir"]:
            out = s["output_dir"]
        elif self.items and self.items[0].path:
            out = os.path.dirname(os.path.abspath(self.items[0].path))
        else:
            out = os.getcwd()
        os.makedirs(out, exist_ok=True)
        self._open_path(out)

    def _open_path(self, path: str):
        try:
            if sys.platform == "win32":
                os.startfile(path)  # noqa
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            QMessageBox.information(self, "Open", f"{path}\n\n({exc})")

    # -------------------------------------------------------------- processing
    def start(self):
        if not self.items:
            QMessageBox.information(self, "Queue", "Add some text files (or text) first.")
            return
        s = self.get_settings()
        if not s["voice"]:
            QMessageBox.information(self, "Voice", "Select a voice first.")
            return
        if s["output_mode"] == "custom" and not s["output_dir"]:
            QMessageBox.information(self, "Output", "Choose an output folder or switch to 'Next to the text file'.")
            return
        if s["format"] != "mp3" and not find_ffmpeg():
            if QMessageBox.question(self, "ffmpeg missing", f"ffmpeg was not found, so {s['format']} cannot be produced; "
                                    "files will be saved as mp3.\n\nContinue?") != QMessageBox.StandardButton.Yes:
                return
        for r in range(self.table.rowCount()):
            self.table.item(r, 1).setText("pending")
            self.table.item(r, 2).setText("")
            self._color_row(r, "pending")
        self.log.clear()
        self._done = self._failed = 0
        self._run_start = time.time()
        self.total_bar.setRange(0, len(self.items))
        self.total_bar.setValue(0)
        self.file_bar.setRange(0, 1)
        self.file_bar.setValue(0)
        self.stat_done.setText("0")
        self.stat_failed.setText("0")
        self._timer.start(1000)
        self.worker = Worker(list(self.items), s)
        w = self.worker
        w.log.connect(self._log)
        w.status.connect(self._on_status)
        w.item_started.connect(self._on_item_started)
        w.item_progress.connect(self._on_item_progress)
        w.item_done.connect(self._on_item_done)
        w.finished_all.connect(self._on_finished)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tabs.setEnabled(False)
        w.start()

    def stop(self):
        if self.worker:
            self.worker.request_stop()
            self.phase_label.setText("Stopping…")
            self.stop_btn.setEnabled(False)

    def _tick(self):
        e = int(time.time() - self._run_start)
        self.stat_elapsed.setText(f"{e // 60}:{e % 60:02d}")

    def _on_status(self, text: str):
        self.phase_label.setText(text)
        self.statusBar().showMessage(text)

    def _on_item_started(self, i: int):
        self.table.item(i, 1).setText("running")
        self._color_row(i, "running")
        self.table.scrollToItem(self.table.item(i, 0))
        self.counter_label.setText(f"item {i + 1} / {len(self.items)} — {self.items[i].title}")
        self.file_bar.setRange(0, 1)
        self.file_bar.setValue(0)

    def _on_item_progress(self, done: int, total: int):
        self.file_bar.setRange(0, max(1, total))
        self.file_bar.setValue(done)

    def _on_item_done(self, i: int, state: str, detail: str):
        self.table.item(i, 1).setText(state)
        self.table.item(i, 2).setText(detail)
        self._color_row(i, state)
        if state == "done":
            self._done += 1
        elif state == "failed":
            self._failed += 1
        self.stat_done.setText(str(self._done))
        self.stat_failed.setText(str(self._failed))
        self.total_bar.setValue(self.total_bar.value() + 1)

    def _on_finished(self, ok: int, total: int):
        self._timer.stop()
        self._tick()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.tabs.setEnabled(True)
        self.phase_label.setText("Finished" if ok == total else "Finished with problems")
        self.counter_label.setText("")
        self.statusBar().showMessage(f"Done: {ok}/{total} items")
        if self.worker:
            self.worker.wait(10000)
            self.worker.deleteLater()
        self.worker = None

    def _log(self, msg: str):
        color = "#c0392b" if msg.startswith("ERROR") else ("#b8860b" if "failed" in msg or "not found" in msg else "#2c3e50")
        safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.log.append(f'<span style="color:{color};">{safe}</span>')
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<b>{APP_NAME} {APP_VERSION}</b><br>Batch text-to-speech with Microsoft Edge neural voices "
                          "(edge-tts). Output mp3 / wav / ogg / flac / m4a with srt / vtt subtitles.<br><br>"
                          "<a href='https://github.com/hclivess/edge-tts-gui'>github.com/hclivess/edge-tts-gui</a>")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    icon = os.path.join(app_dir(), "icon.ico")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))
    win = MainWindow()
    win.show()
    selftest = os.environ.get("EDGE_TTS_GUI_SELFTEST")
    if selftest:
        _selftest(win, selftest)
    sys.exit(app.exec())


def _selftest(win: MainWindow, path: str):
    """Headless smoke test: synthesize one text file to mp3 + srt with the default voice, exit 0 on success"""
    win.apply_settings({"voice": DEFAULT_VOICE, "format": "mp3", "subtitles": "srt", "overwrite": True,
                        "output_mode": "same"})
    win.add_paths([path])

    def finished(ok, total):
        base = os.path.splitext(path)[0]
        good = ok == total and os.path.getsize(base + ".mp3") > 1000 and os.path.getsize(base + ".srt") > 10
        print(f"selftest: finished {ok}/{total} ok={good}", flush=True)
        QTimer.singleShot(0, lambda: (win.worker and win.worker.wait(10000), QApplication.exit(0 if good else 1)))

    def go():
        win.controls["voice"].setEditable(True)
        win.controls["voice"].setCurrentText(DEFAULT_VOICE)
        win.start()
        win.worker.log.connect(lambda m: print("selftest:", m, flush=True))
        win.worker.finished_all.connect(finished)
    QTimer.singleShot(500, go)


if __name__ == "__main__":
    main()
