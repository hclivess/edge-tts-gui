#!/usr/bin/env python3

"""Text-to-Speech GUI application using PySide6 and edge-tts"""

import sys
import asyncio
import tempfile
import os
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox,
    QFileDialog, QProgressBar, QMessageBox, QGroupBox, QSlider,
    QCheckBox, QScrollArea, QSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent
import edge_tts

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class DropZoneTextEdit(QTextEdit):
    """Custom QTextEdit that supports drag and drop of text files"""
    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("Drop text file here or paste text...")
        self.update_style(False)

    def update_style(self, hover):
        border_color = "#0078d4" if hover else "#bbb"
        bg_color = "#e8f4fd" if hover else "white"
        self.setStyleSheet(f"""
            QTextEdit {{
                border: 2px dashed {border_color};
                border-radius: 8px;
                padding: 8px;
                background-color: {bg_color};
                font-size: 12px;
                font-family: 'Segoe UI', Arial;
            }}
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._is_valid_drop(event.mimeData()):
            event.acceptProposedAction()
            self.update_style(True)

    def dragLeaveEvent(self, event):
        self.update_style(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        self.update_style(False)
        mime = event.mimeData()
        
        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if self._is_text_file(path):
                        self.file_dropped.emit(path)
                        event.acceptProposedAction()
                        return
        
        if mime.hasText():
            self.setPlainText(mime.text())
            event.acceptProposedAction()

    def _is_valid_drop(self, mime_data):
        if mime_data.hasText():
            return True
        if mime_data.hasUrls():
            for url in mime_data.urls():
                if url.isLocalFile() and self._is_text_file(url.toLocalFile()):
                    return True
        return False

    def _is_text_file(self, path):
        text_extensions = ['.txt', '.md', '.text', '.rst', '.log']
        return any(path.lower().endswith(ext) for ext in text_extensions)


class TextSplitter:
    """Enhanced text splitter with multiple splitting strategies"""
    
    @staticmethod
    def split_text(text, max_chunk_size=3000, method="Sentences"):
        """
        Split text into chunks based on the selected method
        
        Args:
            text: Input text to split
            max_chunk_size: Maximum size of each chunk
            method: Splitting method - "Sentences", "Paragraphs", "Lines", or "None"
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        if method == "None":
            return [text]  # Don't split, let edge-tts handle it
        elif method == "Paragraphs":
            return TextSplitter._split_by_paragraphs(text, max_chunk_size)
        elif method == "Lines":
            return TextSplitter._split_by_lines(text, max_chunk_size)
        else:  # "Sentences" (default)
            return TextSplitter._split_by_sentences(text, max_chunk_size)
    
    @staticmethod
    def _split_by_sentences(text, max_chunk_size):
        """Split text at sentence boundaries (periods)"""
        import re
        
        chunks = []
        current_chunk = ""
        
        # Split by periods, keeping the period with each sentence
        period_pattern = r'(?<![\.])\.(?![\.])(?=\s|$)'
        segments = re.split(period_pattern, text)
        
        # Reconstruct sentences with periods
        sentences = []
        for i, segment in enumerate(segments):
            segment = segment.strip()
            if segment:
                if i < len(segments) - 1:
                    sentences.append(segment + ".")
                else:
                    sentences.append(segment)
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, but keep it as-is
                    chunks.append(sentence.strip())
                    current_chunk = ""
            else:
                current_chunk = potential_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def _split_by_paragraphs(text, max_chunk_size):
        """Split text at paragraph boundaries (double newlines)"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Single paragraph is too long, try splitting by sentences
                    if len(paragraph) > max_chunk_size:
                        sub_chunks = TextSplitter._split_by_sentences(paragraph, max_chunk_size)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        chunks.append(paragraph.strip())
                        current_chunk = ""
            else:
                current_chunk = potential_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def _split_by_lines(text, max_chunk_size):
        """Split text at line boundaries (single newlines)"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        chunks = []
        current_chunk = ""
        
        for line in lines:
            potential_chunk = current_chunk + "\n" + line if current_chunk else line
            
            if len(potential_chunk) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = line
                else:
                    # Single line is too long
                    if len(line) > max_chunk_size:
                        sub_chunks = TextSplitter._split_by_sentences(line, max_chunk_size)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        chunks.append(line.strip())
                        current_chunk = ""
            else:
                current_chunk = potential_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class VoiceCache:
    """Manage voice cache storage"""
    
    @staticmethod
    def get_cache_file():
        """Get path to voice cache file"""
        cache_dir = Path.home() / ".tts_generator"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / "voices_cache.json"
    
    @staticmethod
    def load_voices():
        """Load voices from cache"""
        cache_file = VoiceCache.get_cache_file()
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    @staticmethod
    def save_voices(voices):
        """Save voices to cache"""
        cache_file = VoiceCache.get_cache_file()
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(voices, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False


class TTSWorker(QThread):
    """Worker thread for TTS generation with retry mechanism and text splitting"""
    progress = Signal(str)
    finished = Signal(bool, str)
    retry_attempt = Signal(int, int)  # current_attempt, max_attempts

    def __init__(self, text, voice, output_file, rate, volume, pitch, 
                 generate_subtitles, split_method, max_retries=3, retry_delay=2):
        super().__init__()
        self.text = text
        self.voice = voice
        self.output_file = output_file
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.generate_subtitles = generate_subtitles
        self.split_method = split_method
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temp_files = []

    def run(self):
        # Check if we need to split the text
        if self.split_method != "None" and len(self.text) > 3000:
            self.progress.emit(f"Text is long, splitting by {self.split_method.lower()}...")
            chunks = TextSplitter.split_text(self.text, max_chunk_size=3000, method=self.split_method)
            self.progress.emit(f"Split into {len(chunks)} chunks")
            
            if len(chunks) > 1:
                result = self._generate_with_chunks(chunks)
                self.finished.emit(result[0], result[1])
                return
        
        # Single chunk generation
        for attempt in range(self.max_retries):
            try:
                self.retry_attempt.emit(attempt + 1, self.max_retries)
                self.progress.emit(f"Starting attempt {attempt + 1}/{self.max_retries}...")
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    if self.generate_subtitles and len(self.text) <= 3000:
                        result_msg = loop.run_until_complete(self._generate_with_subtitles())
                    else:
                        if self.generate_subtitles and len(self.text) > 3000:
                            self.progress.emit("Subtitles disabled for long texts")
                        result_msg = loop.run_until_complete(self._generate_audio_only())
                    
                    self.progress.emit("Complete!")
                    self.finished.emit(True, result_msg)
                    return
                    
                finally:
                    self._cleanup_loop(loop)
                    
            except Exception as e:
                error_msg = str(e)
                self.progress.emit(f"Attempt {attempt + 1} failed: {error_msg}")
                
                if attempt < self.max_retries - 1:
                    self.progress.emit(f"Waiting {self.retry_delay} seconds before retry...")
                    self.msleep(self.retry_delay * 1000)
                else:
                    self.finished.emit(False, f"Failed after {self.max_retries} attempts: {error_msg}")

    def _generate_with_chunks(self, chunks):
        """Generate audio for multiple text chunks and combine them"""
        try:
            audio_segments = []
            
            for i, chunk in enumerate(chunks):
                self.progress.emit(f"Processing chunk {i+1}/{len(chunks)}...")
                
                # Generate temporary file for this chunk
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file_path = temp_file.name
                temp_file.close()
                self.temp_files.append(temp_file_path)
                
                # Generate audio for chunk with retry
                chunk_success = False
                for attempt in range(self.max_retries):
                    try:
                        self.retry_attempt.emit(attempt + 1, self.max_retries)
                        self.progress.emit(f"Chunk {i+1}, attempt {attempt + 1}/{self.max_retries}")
                        
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        try:
                            communicate = edge_tts.Communicate(
                                chunk, self.voice,
                                rate=self.rate, volume=self.volume, pitch=self.pitch
                            )
                            loop.run_until_complete(communicate.save(temp_file_path))
                            chunk_success = True
                            break
                            
                        finally:
                            self._cleanup_loop(loop)
                            
                    except Exception as e:
                        self.progress.emit(f"Chunk {i+1} attempt {attempt + 1} failed: {str(e)}")
                        if attempt < self.max_retries - 1:
                            self.msleep(self.retry_delay * 1000)
                
                if not chunk_success:
                    self.progress.emit(f"Skipping chunk {i+1} due to repeated failures")
                    continue
            
            # Collect successful chunks
            successful_chunks = []
            for temp_file in self.temp_files:
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                    successful_chunks.append(temp_file)
            
            if len(successful_chunks) == 0:
                return False, "Failed to generate any audio chunks"
            elif len(successful_chunks) == 1:
                # Only one chunk, just rename it
                os.rename(successful_chunks[0], self.output_file)
                result_msg = f"Audio saved: {self.output_file} (1 chunk)"
            else:
                # Multiple chunks - combine if pydub available
                if PYDUB_AVAILABLE:
                    self.progress.emit("Combining audio chunks...")
                    combined = AudioSegment.empty()
                    for i, temp_file in enumerate(successful_chunks):
                        self.progress.emit(f"Adding chunk {i+1}/{len(successful_chunks)}...")
                        try:
                            audio_segment = AudioSegment.from_mp3(temp_file)
                            combined += audio_segment
                        except Exception as e:
                            self.progress.emit(f"Error loading chunk {i+1}: {str(e)}")
                            continue
                    
                    if len(combined) > 0:
                        combined.export(self.output_file, format="mp3")
                        result_msg = f"Audio saved: {self.output_file} ({len(successful_chunks)} chunks combined)"
                    else:
                        return False, "Failed to combine audio chunks"
                else:
                    # Without pydub, save separate files
                    base_name = self.output_file.rsplit('.', 1)[0]
                    extension = self.output_file.rsplit('.', 1)[1] if '.' in self.output_file else 'mp3'
                    
                    for i, temp_file in enumerate(successful_chunks):
                        chunk_output = f"{base_name}_part{i+1:02d}.{extension}"
                        os.rename(temp_file, chunk_output)
                    
                    result_msg = f"Audio saved as {len(successful_chunks)} files: {base_name}_part01.{extension}, etc.\n(Install pydub to combine automatically)"
            
            return True, result_msg
            
        except Exception as e:
            return False, f"Error processing chunks: {str(e)}"
        finally:
            self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Clean up temporary audio files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self.temp_files = []

    def _cleanup_loop(self, loop):
        """Clean up event loop properly"""
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        except Exception:
            pass

    async def _generate_audio_only(self):
        communicate = edge_tts.Communicate(
            self.text, self.voice,
            rate=self.rate, volume=self.volume, pitch=self.pitch
        )
        await communicate.save(self.output_file)
        return f"Audio saved: {self.output_file}"

    async def _generate_with_subtitles(self):
        communicate = edge_tts.Communicate(
            self.text, self.voice,
            rate=self.rate, volume=self.volume, pitch=self.pitch
        )

        subtitle_file = self.output_file.rsplit('.', 1)[0] + '.vtt'
        submaker = edge_tts.SubMaker()

        with open(self.output_file, 'wb') as audio_fp:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_fp.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

        with open(subtitle_file, 'w', encoding='utf-8') as subtitle_fp:
            subtitle_fp.write(submaker.generate_subs())

        return f"Audio: {self.output_file}\nSubtitles: {subtitle_file}"


class VoiceLoader(QThread):
    """Worker thread for loading voices with retry"""
    finished = Signal(list)
    progress = Signal(str)
    retry_attempt = Signal(int, int)

    def __init__(self, max_retries=3, retry_delay=2):
        super().__init__()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def run(self):
        for attempt in range(self.max_retries):
            try:
                self.retry_attempt.emit(attempt + 1, self.max_retries)
                self.progress.emit(f"Loading voices... (attempt {attempt + 1}/{self.max_retries})")
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    voices = loop.run_until_complete(asyncio.wait_for(edge_tts.list_voices(), timeout=10.0))
                    if voices:
                        self.progress.emit("Voices loaded successfully!")
                        self.finished.emit(voices)
                        return
                finally:
                    self._cleanup_loop(loop)
                    
            except Exception as e:
                self.progress.emit(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    self.msleep(self.retry_delay * 1000)

        self.progress.emit("Failed to load voices after all retries")
        self.finished.emit([])

    def _cleanup_loop(self, loop):
        """Clean up event loop properly"""
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        except Exception:
            pass


class TTSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTS Generator")
        self.setMinimumSize(900, 800)
        self.resize(950, 850)

        self.output_file = ""
        self.worker = None
        self.voices = []
        self._voices_loaded = False
        self.voice_loader = None

        self.settings = QSettings("TTSGenerator", "TTSApp")

        # Default retry settings
        self.max_retries = 3
        self.retry_delay = 2

        self.setup_ui()
        self.load_settings()
        self.load_voices()

    def closeEvent(self, event):
        """Clean up threads when closing window"""
        for thread in [self.worker, self.voice_loader]:
            if thread is not None and thread.isRunning():
                thread.terminate()
                thread.wait(1000)
        event.accept()

    def setup_ui(self):
        """Setup modern UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        scroll_widget = QWidget()
        scroll.setWidget(scroll_widget)

        main_layout = QVBoxLayout(scroll_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(20, 20, 20, 20)

        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        # Title
        title = QLabel("🎤 Text-to-Speech Generator")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; padding: 10px;")
        main_layout.addWidget(title)

        # Text input
        text_group = self._create_group_box("📝 Text Input")
        text_layout = QVBoxLayout()
        
        self.text_preview = DropZoneTextEdit()
        self.text_preview.setMinimumHeight(150)
        self.text_preview.setMaximumHeight(200)
        self.text_preview.file_dropped.connect(self.handle_file_drop)
        self.text_preview.textChanged.connect(self.update_char_count)
        
        self.char_count_label = QLabel("Characters: 0")
        self.char_count_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")

        text_layout.addWidget(self.text_preview)
        text_layout.addWidget(self.char_count_label)
        text_group.setLayout(text_layout)
        main_layout.addWidget(text_group)

        # Voice selection
        voice_group = self._create_group_box("🗣️ Voice")
        voice_layout = QVBoxLayout()
        
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self.language_filter = QComboBox()
        self.language_filter.addItem("All Languages")
        self.language_filter.currentTextChanged.connect(self.filter_voices)
        lang_row.addWidget(self.language_filter, 1)
        
        self.refresh_voices_btn = self._create_button("🔄 Refresh", "#3498db", self.refresh_voices)
        self.refresh_voices_btn.setMaximumWidth(100)
        lang_row.addWidget(self.refresh_voices_btn)

        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItem("Loading...")
        self.voice_combo.currentIndexChanged.connect(self.save_settings)
        voice_row.addWidget(self.voice_combo, 1)

        voice_layout.addLayout(lang_row)
        voice_layout.addLayout(voice_row)
        voice_group.setLayout(voice_layout)
        main_layout.addWidget(voice_group)

        # Voice customization
        custom_group = self._create_group_box("⚙️ Voice Customization")
        custom_layout = QVBoxLayout()
        
        self.rate_slider, self.rate_value = self._create_slider_with_label("Rate:", -100, 100, 0, "%")
        self.volume_slider, self.volume_value = self._create_slider_with_label("Volume:", -100, 100, 0, "%")
        self.pitch_slider, self.pitch_value = self._create_slider_with_label("Pitch:", -100, 100, 0, "Hz")
        
        custom_layout.addLayout(self._create_slider_layout(self.rate_slider, self.rate_value, "Rate:"))
        custom_layout.addLayout(self._create_slider_layout(self.volume_slider, self.volume_value, "Volume:"))
        custom_layout.addLayout(self._create_slider_layout(self.pitch_slider, self.pitch_value, "Pitch:"))
        
        reset_btn = self._create_button("Reset All", "#95a5a6", self.reset_all)
        
        reset_layout = QHBoxLayout()
        reset_layout.addStretch()
        reset_layout.addWidget(reset_btn)
        custom_layout.addLayout(reset_layout)
        
        custom_group.setLayout(custom_layout)
        main_layout.addWidget(custom_group)

        # Text splitting options
        split_group = self._create_group_box("✂️ Text Splitting (for long texts >3000 chars)")
        split_layout = QHBoxLayout()
        
        split_layout.addWidget(QLabel("Split Method:"))
        self.split_method_combo = QComboBox()
        self.split_method_combo.addItems(["Sentences", "Paragraphs", "Lines", "None"])
        self.split_method_combo.setCurrentText("Sentences")
        self.split_method_combo.setToolTip(
            "Sentences: Split at periods (.)\n"
            "Paragraphs: Split at double newlines\n"
            "Lines: Split at single newlines\n"
            "None: Don't split (may fail for very long texts)"
        )
        self.split_method_combo.currentTextChanged.connect(self.save_settings)
        self.split_method_combo.currentTextChanged.connect(self.update_char_count)
        split_layout.addWidget(self.split_method_combo, 1)
        
        split_group.setLayout(split_layout)
        main_layout.addWidget(split_group)

        # Retry settings
        retry_group = self._create_group_box("🔄 Retry Settings")
        retry_layout = QVBoxLayout()
        
        retry_count_layout = QHBoxLayout()
        retry_count_layout.addWidget(QLabel("Max Retries:"))
        self.max_retries_spin = QSpinBox()
        self.max_retries_spin.setRange(1, 10)
        self.max_retries_spin.setValue(3)
        self.max_retries_spin.valueChanged.connect(self.update_retry_settings)
        retry_count_layout.addWidget(self.max_retries_spin)
        retry_count_layout.addStretch()
        
        retry_delay_layout = QHBoxLayout()
        retry_delay_layout.addWidget(QLabel("Retry Delay (sec):"))
        self.retry_delay_spin = QSpinBox()
        self.retry_delay_spin.setRange(1, 10)
        self.retry_delay_spin.setValue(2)
        self.retry_delay_spin.valueChanged.connect(self.update_retry_settings)
        retry_delay_layout.addWidget(self.retry_delay_spin)
        retry_delay_layout.addStretch()
        
        retry_layout.addLayout(retry_count_layout)
        retry_layout.addLayout(retry_delay_layout)
        retry_group.setLayout(retry_layout)
        main_layout.addWidget(retry_group)

        # Output
        output_group = self._create_group_box("💾 Output")
        output_layout = QVBoxLayout()
        
        file_row = QHBoxLayout()
        self.output_line = QLineEdit()
        self.output_line.setPlaceholderText("output.mp3")
        self.output_line.textChanged.connect(self.save_settings)
        browse_btn = self._create_button("Browse", "#3498db", self.browse_output)
        
        file_row.addWidget(self.output_line)
        file_row.addWidget(browse_btn)

        self.subtitle_check = QCheckBox("Generate subtitles (.vtt) - only for texts <3000 chars")
        self.subtitle_check.setStyleSheet("font-size: 11px;")
        self.subtitle_check.stateChanged.connect(self.save_settings)

        output_layout.addLayout(file_row)
        output_layout.addWidget(self.subtitle_check)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)

        # Generate button
        self.generate_btn = QPushButton("🎬 Generate Audio")
        self.generate_btn.setMinimumHeight(45)
        self.generate_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.generate_btn.clicked.connect(self.generate_audio)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover { background-color: #229954; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        main_layout.addWidget(self.generate_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #ecf0f1;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 2px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumWidth(500)
        self.status_label.setMaximumHeight(40)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #7f8c8d; 
                font-size: 11px; 
                padding: 5px;
                background-color: #f8f9fa;
                border-radius: 4px;
                border: 1px solid #e9ecef;
            }
        """)
        main_layout.addWidget(self.status_label)

        # Retry status label
        self.retry_status_label = QLabel("")
        self.retry_status_label.setAlignment(Qt.AlignCenter)
        self.retry_status_label.setMinimumWidth(500)
        self.retry_status_label.setMaximumHeight(30)
        self.retry_status_label.setStyleSheet("""
            QLabel {
                color: #e67e22; 
                font-size: 11px; 
                padding: 3px;
                background-color: #fef9e7;
                border-radius: 4px;
                border: 1px solid #fdebd0;
                font-weight: bold;
            }
        """)
        main_layout.addWidget(self.retry_status_label)

        main_layout.addStretch()

    def save_settings(self):
        """Save all settings to persistent storage"""
        self.settings.setValue("rate", self.rate_slider.value())
        self.settings.setValue("volume", self.volume_slider.value())
        self.settings.setValue("pitch", self.pitch_slider.value())
        self.settings.setValue("max_retries", self.max_retries)
        self.settings.setValue("retry_delay", self.retry_delay)
        self.settings.setValue("output_file", self.output_line.text())
        self.settings.setValue("generate_subtitles", self.subtitle_check.isChecked())
        self.settings.setValue("selected_voice", self.voice_combo.currentText())
        self.settings.setValue("selected_language", self.language_filter.currentText())
        self.settings.setValue("split_method", self.split_method_combo.currentText())
    
    def load_settings(self):
        """Load all settings from persistent storage"""
        self.rate_slider.setValue(self.settings.value("rate", 0, type=int))
        self.volume_slider.setValue(self.settings.value("volume", 0, type=int))
        self.pitch_slider.setValue(self.settings.value("pitch", 0, type=int))
        
        self.max_retries = self.settings.value("max_retries", 3, type=int)
        self.retry_delay = self.settings.value("retry_delay", 2, type=int)
        self.max_retries_spin.setValue(self.max_retries)
        self.retry_delay_spin.setValue(self.retry_delay)
        
        output_file = self.settings.value("output_file", "output_audio.mp3", type=str)
        self.output_line.setText(output_file)
        
        generate_subs = self.settings.value("generate_subtitles", False, type=bool)
        self.subtitle_check.setChecked(generate_subs)
        
        split_method = self.settings.value("split_method", "Sentences", type=str)
        idx = self.split_method_combo.findText(split_method)
        if idx >= 0:
            self.split_method_combo.setCurrentIndex(idx)

    def update_retry_settings(self):
        """Update retry settings from UI controls"""
        self.max_retries = self.max_retries_spin.value()
        self.retry_delay = self.retry_delay_spin.value()
        self.save_settings()

    def _create_group_box(self, title):
        """Create a styled group box"""
        group = QGroupBox(title)
        group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        return group

    def _create_slider_with_label(self, label, min_val, max_val, default, suffix):
        """Create a slider and its value label"""
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        
        value_label = QLabel(f"{default:+d}{suffix}")
        value_label.setMinimumWidth(40)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        if suffix == "%":
            slider.valueChanged.connect(lambda v: value_label.setText(f"{v:+d}%"))
        else:
            slider.valueChanged.connect(lambda v: value_label.setText(f"{v:+d}{suffix}"))
        
        slider.valueChanged.connect(self.save_settings)
        slider.valueChanged.connect(self.update_char_count)
        
        return slider, value_label

    def _create_slider_layout(self, slider, value_label, label_text):
        """Create layout for slider with value label"""
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setMinimumWidth(50)
        
        layout.addWidget(label)
        layout.addWidget(slider, 1)
        layout.addWidget(value_label)
        return layout

    def _create_button(self, text, color, callback):
        """Create a styled button"""
        button = QPushButton(text)
        button.setMaximumWidth(100)
        button.clicked.connect(callback)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px;
            }}
            QPushButton:hover {{ background-color: #7f8c8d; }}
        """)
        return button

    def reset_all(self):
        """Reset all sliders to default"""
        self.rate_slider.setValue(0)
        self.volume_slider.setValue(0)
        self.pitch_slider.setValue(0)
        self.save_settings()

    def update_char_count(self):
        """Update character count and duration estimate"""
        text = self.text_preview.toPlainText()
        char_count = len(text)
        word_count = len(text.split())

        # Estimate duration: ~150 words/min at normal speed
        rate_multiplier = 1.0 + (self.rate_slider.value() / 100.0)
        base_wpm = 150
        adjusted_wpm = base_wpm * rate_multiplier

        estimated_minutes = word_count / adjusted_wpm if adjusted_wpm > 0 else 0
        duration_str = f"{int(estimated_minutes * 60)}s" if estimated_minutes < 1 else f"{estimated_minutes:.1f}min"

        # Check if text will be split
        split_method = self.split_method_combo.currentText()
        will_split = split_method != "None" and char_count > 3000

        status_parts = [
            f"Characters: {char_count:,}", 
            f"Words: {word_count:,}", 
            f"~{duration_str}"
        ]

        if will_split:
            chunks = TextSplitter.split_text(text, method=split_method)
            status_parts.append(f"⚠️ Will split into {len(chunks)} chunks ({split_method.lower()})")
            self.subtitle_check.setChecked(False)
            self.subtitle_check.setEnabled(False)
            self.char_count_label.setStyleSheet("color: #e67e22; font-size: 11px; font-weight: bold;")
        else:
            self.subtitle_check.setEnabled(True)
            self.char_count_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")

        self.char_count_label.setText(" | ".join(status_parts))

    def handle_file_drop(self, file_path):
        """Handle dropped text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.text_preview.setPlainText(content)
            self.status_label.setText(f"Loaded: {Path(file_path).name}")
            self.status_label.setStyleSheet("color: #27ae60; font-size: 11px; padding: 5px;")
        except Exception as e:
            self.show_error(f"Error loading file: {str(e)}")

    def load_voices(self, force_refresh=False):
        """Load voices from cache or network"""
        if not force_refresh:
            cached_voices = VoiceCache.load_voices()
            if cached_voices:
                self.voices = cached_voices
                self._voices_loaded = True
                self.populate_voices()
                self.restore_voice_selection()
                self.status_label.setText(f"✅ Loaded {len(cached_voices)} cached voices")
                self.status_label.setStyleSheet("color: #27ae60; font-size: 11px; padding: 5px;")
                return

        self.status_label.setText("Loading voices from network...")
        self.status_label.setStyleSheet("color: #3498db; font-size: 11px; padding: 5px;")
        self.retry_status_label.setText("")
        self.voice_combo.setEnabled(False)
        self.language_filter.setEnabled(False)
        self.refresh_voices_btn.setEnabled(False)

        self.voice_loader = VoiceLoader(max_retries=self.max_retries, retry_delay=self.retry_delay)
        self.voice_loader.progress.connect(self.update_voice_loading_status)
        self.voice_loader.retry_attempt.connect(self.update_retry_display)
        self.voice_loader.finished.connect(self.voices_loaded)
        self.voice_loader.start()
    
    def refresh_voices(self):
        """Force refresh voices from network"""
        reply = QMessageBox.question(
            self, "Refresh Voices",
            "This will download the latest voice list from the network. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.load_voices(force_refresh=True)

    def update_voice_loading_status(self, msg):
        """Update status during voice loading"""
        self.status_label.setText(msg)

    def update_retry_display(self, current_attempt, max_attempts):
        """Update retry status display"""
        self.retry_status_label.setText(f"🔄 Attempt {current_attempt} of {max_attempts}")

    def voices_loaded(self, voices):
        """Handle loaded voices and store them"""
        self.retry_status_label.setText("")
        self.refresh_voices_btn.setEnabled(True)

        if voices:
            self.voices = voices
            self._voices_loaded = True
            
            cache_msg = " (cached)" if VoiceCache.save_voices(voices) else " (cache failed)"
            
            self.populate_voices()
            self.restore_voice_selection()
            self.voice_combo.setEnabled(True)
            self.language_filter.setEnabled(True)
            self.status_label.setText(f"✅ Loaded {len(voices)} voices{cache_msg}")
            self.status_label.setStyleSheet("color: #27ae60; font-size: 11px; padding: 5px;")
        else:
            self.voice_combo.clear()
            self.voice_combo.addItem("Failed to load voices")
            self.status_label.setText("❌ Failed to load voices - Click 🔄 Refresh to retry")
            self.status_label.setStyleSheet("color: #e74c3c; font-size: 11px; padding: 5px;")
            self.voice_combo.setEnabled(False)
            self.language_filter.setEnabled(False)
    
    def restore_voice_selection(self):
        """Restore previously selected voice and language"""
        saved_language = self.settings.value("selected_language", "All Languages", type=str)
        idx = self.language_filter.findText(saved_language)
        if idx >= 0:
            self.language_filter.setCurrentIndex(idx)
        
        saved_voice = self.settings.value("selected_voice", "", type=str)
        if saved_voice:
            idx = self.voice_combo.findText(saved_voice)
            if idx >= 0:
                self.voice_combo.setCurrentIndex(idx)

    def filter_voices(self):
        """Filter voices based on selected language"""
        self.populate_voices()
        self.save_settings()

    def populate_voices(self):
        """Populate voice combo with stored voices"""
        if not self.voices:
            return

        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()

        selected_lang = self.language_filter.currentText()
        filtered_voices = []

        for voice in self.voices:
            locale = voice.get('Locale', '')
            if selected_lang == "All Languages" or locale.startswith(selected_lang.lower()):
                filtered_voices.append(voice)

        filtered_voices.sort(key=lambda v: v.get('ShortName', ''))

        if self.language_filter.count() == 1:
            languages = set()
            for voice in self.voices:
                locale = voice.get('Locale', '')
                if locale:
                    languages.add(locale.split('-')[0].upper())
            
            for lang in sorted(languages):
                self.language_filter.addItem(lang)

        for voice in filtered_voices:
            name = voice.get('ShortName', 'Unknown')
            gender = voice.get('Gender', '')
            locale = voice.get('Locale', '')
            display = f"{name} ({locale} - {gender})"
            self.voice_combo.addItem(display, name)

        if selected_lang == "CS" or (selected_lang == "All Languages" and self.language_filter.findText("CS") >= 0):
            for i in range(self.voice_combo.count()):
                if "AntoninNeural" in self.voice_combo.itemText(i):
                    self.voice_combo.setCurrentIndex(i)
                    break
        elif self.voice_combo.count() > 0:
            self.voice_combo.setCurrentIndex(0)
        
        self.voice_combo.blockSignals(False)

    def browse_output(self):
        """Browse for output file location"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "output_audio.mp3", "MP3 (*.mp3);;All (*.*)"
        )
        if path:
            self.output_line.setText(path)

    def get_voice_id(self):
        """Get selected voice ID"""
        idx = self.voice_combo.currentIndex()
        return self.voice_combo.itemData(idx) if idx >= 0 else None

    def generate_audio(self):
        """Start audio generation with retry mechanism"""
        text = self.text_preview.toPlainText().strip()
        if not text:
            self.show_error("Please enter text")
            return

        output = self.output_line.text()
        if not output:
            self.show_error("Please specify output file")
            return

        voice = self.get_voice_id()
        if not voice:
            self.show_error("Please select a voice")
            return

        if self.worker is not None and self.worker.isRunning():
            self.show_error("Generation already in progress!")
            return

        rate = f"{self.rate_slider.value():+d}%"
        volume = f"{self.volume_slider.value():+d}%"
        pitch = f"{self.pitch_slider.value():+d}Hz"
        split_method = self.split_method_combo.currentText()

        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.retry_status_label.setText("")

        self.worker = TTSWorker(
            text, voice, output, rate, volume, pitch,
            self.subtitle_check.isChecked(),
            split_method,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay
        )
        self.worker.progress.connect(self.update_status)
        self.worker.retry_attempt.connect(self.update_retry_display)
        self.worker.finished.connect(self.generation_finished)
        self.worker.start()

    def update_status(self, msg):
        """Update status message"""
        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: #3498db; font-size: 11px; padding: 5px;")

    def generation_finished(self, success, msg):
        """Handle generation completion"""
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.retry_status_label.setText("")

        if self.worker:
            if self.worker.isRunning():
                self.worker.terminate()
            self.worker.wait(1000)
            self.worker = None

        if success:
            self.status_label.setText(msg)
            self.status_label.setStyleSheet("color: #27ae60; font-size: 11px; padding: 5px;")
            QMessageBox.information(self, "Success", msg)
        else:
            self.status_label.setText("Failed")
            self.status_label.setStyleSheet("color: #e74c3c; font-size: 11px; padding: 5px;")
            self.show_error(msg)

    def show_error(self, msg):
        """Show error message"""
        QMessageBox.critical(self, "Error", msg)


def main():
    app = QApplication(sys.argv)
    window = TTSMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()