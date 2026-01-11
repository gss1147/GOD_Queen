import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from PyQt5 import QtWidgets, QtCore, QtGui

from Hope_DuGan_1 import HopeAssistant
from generation import GenerationCore
from pdf_evolution import PDFEvolutionCore
from Spiritual import daily_spiritual_message, tarot_reading
from GOD_Agents import GodAgentsCore
from memory import MemoryCore, MemoryRecord


class UniversalChatManager(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: List[QtWidgets.QTextEdit] = []

    def register(self, widget: QtWidgets.QTextEdit) -> None:
        if widget not in self._widgets:
            self._widgets.append(widget)

    def append_line(self, text: str) -> None:
        for w in self._widgets:
            w.append(text)
            cursor = w.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            w.setTextCursor(cursor)


class ChatCoreTab(QtWidgets.QWidget):
    def __init__(self, universal_chat: UniversalChatManager, parent=None):
        super().__init__(parent)
        self.universal_chat = universal_chat
        self.assistant = HopeAssistant()
        self._build_ui()

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        self.history_list = QtWidgets.QListWidget()
        main_layout.addWidget(self.history_list, 1)

        right_panel = QtWidgets.QVBoxLayout()

        header = QtWidgets.QLabel("THY HOPE DuGAN")
        header.setAlignment(QtCore.Qt.AlignCenter)
        header.setStyleSheet("font-size: 20pt; font-weight: bold;")
        right_panel.addWidget(header)

        self.chat_box = QtWidgets.QTextEdit()
        self.chat_box.setReadOnly(True)
        self.universal_chat.register(self.chat_box)
        right_panel.addWidget(self.chat_box, 4)

        self.input_box = QtWidgets.QTextEdit()
        self.input_box.setPlaceholderText("Type your message to Hope...")
        right_panel.addWidget(self.input_box, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_voice = QtWidgets.QPushButton("Realtime VOICE-CHAT")
        self.btn_load = QtWidgets.QPushButton("LOAD")
        self.btn_send = QtWidgets.QPushButton("SEND")
        btn_row.addWidget(self.btn_voice)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_send)
        right_panel.addLayout(btn_row)

        main_layout.addLayout(right_panel, 3)

        self.btn_send.clicked.connect(self._on_send)
        self.btn_load.clicked.connect(self._on_load)
        self.btn_voice.clicked.connect(self._on_voice_chat)

    def _on_send(self):
        text = self.input_box.toPlainText().strip()
        if not text:
            return
        self.input_box.clear()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.history_list.addItem(f"[{ts}] YOU: {text}")
        self.universal_chat.append_line(f"YOU: {text}")
        reply = self.assistant.chat(text)
        self.history_list.addItem(f"[{ts}] HOPE: {reply}")
        self.universal_chat.append_line(f"HOPE: {reply}")

    def _on_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Text File", "", "Text Files (*.txt);;All Files (*.*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load file:\n{e}")
            return
        self.input_box.setPlainText(content)

    def _on_voice_chat(self):
        QtWidgets.QMessageBox.information(
            self,
            "Voice Chat",
            "Real-time voice chat is not wired yet. Use text chat for now.",
        )


class GenerationalCoreTab(QtWidgets.QWidget):
    def __init__(self, universal_chat: UniversalChatManager, parent=None):
        super().__init__(parent)
        self.universal_chat = universal_chat
        self.core = GenerationCore()
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel("THY CREATOR")
        header.setAlignment(QtCore.Qt.AlignCenter)
        header.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(header)

        self.chat_box = QtWidgets.QTextEdit()
        self.chat_box.setReadOnly(True)
        self.chat_box.setAcceptDrops(True)
        self.universal_chat.register(self.chat_box)
        layout.addWidget(self.chat_box, 4)

        input_row = QtWidgets.QHBoxLayout()
        self.input_box = QtWidgets.QTextEdit()
        self.input_box.setPlaceholderText("Describe what you want to generate...")
        self.btn_send = QtWidgets.QPushButton("SEND")
        input_row.addWidget(self.input_box, 4)
        input_row.addWidget(self.btn_send, 1)
        layout.addLayout(input_row)

        row1 = QtWidgets.QHBoxLayout()
        self.btn_regen = QtWidgets.QPushButton("REGENERATE")
        self.btn_debug = QtWidgets.QPushButton("DEBUG")
        self.btn_test = QtWidgets.QPushButton("TEST")
        self.btn_fix = QtWidgets.QPushButton("FIX")
        self.btn_facts = QtWidgets.QPushButton("Facts")
        self.btn_generate = QtWidgets.QPushButton("GENERATE")
        self.btn_load = QtWidgets.QPushButton("LOAD")
        for b in (
            self.btn_regen,
            self.btn_debug,
            self.btn_test,
            self.btn_fix,
            self.btn_facts,
            self.btn_generate,
            self.btn_load,
        ):
            row1.addWidget(b)
        layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self.btn_selfie = QtWidgets.QPushButton("GOD QUEEN Selfie")
        self.btn_video = QtWidgets.QPushButton("GOD QUEEN Video")
        row2.addWidget(self.btn_selfie)
        row2.addWidget(self.btn_video)
        layout.addLayout(row2)

        self.btn_send.clicked.connect(self._on_send)
        self.btn_generate.clicked.connect(self._on_generate_generic)
        self.btn_load.clicked.connect(self._on_load_file)
        self.btn_selfie.clicked.connect(self._on_selfie)
        self.btn_video.clicked.connect(self._on_video)

    def _get_prompt(self) -> str:
        return self.input_box.toPlainText().strip()

    def _on_send(self):
        prompt = self._get_prompt()
        if not prompt:
            return
        self.universal_chat.append_line(f"GEN REQUEST: {prompt}")

    def _on_generate_generic(self):
        prompt = self._get_prompt()
        if not prompt:
            return
        path = self.core.generate_text_code(prompt)
        self.universal_chat.append_line(f"Generated code saved to: {path}")

    def _on_load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load File", "", "All Files (*.*)"
        )
        if not path:
            return
        self.universal_chat.append_line(f"Loaded file for context: {path}")

    def _on_selfie(self):
        base_prompt = "cinematic portrait of a woman in a white silk suit, ultra-detailed, dramatic lighting"
        path = self.core.generate_image(base_prompt)
        if path:
            self.universal_chat.append_line(f"GOD QUEEN Selfie generated at: {path}")
        else:
            self.universal_chat.append_line("Image generation not available.")

    def _on_video(self):
        self.universal_chat.append_line("Video generation requires an image sequence configured in GenerationCore.")


class SpiritualCoreTab(QtWidgets.QWidget):
    def __init__(self, universal_chat: UniversalChatManager, parent=None):
        super().__init__(parent)
        self.universal_chat = universal_chat
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel("THY SPIRITUAL REALM")
        header.setAlignment(QtCore.Qt.AlignCenter)
        header.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(header)

        self.chat_box = QtWidgets.QTextEdit()
        self.chat_box.setReadOnly(True)
        self.universal_chat.register(self.chat_box)
        layout.addWidget(self.chat_box, 4)

        btn_row = QtWidgets.QHBoxLayout()
        self.intent_edit = QtWidgets.QLineEdit()
        self.intent_edit.setPlaceholderText("Intent / focus for daily message...")
        self.btn_daily = QtWidgets.QPushButton("Daily Message")
        self.tarot_question = QtWidgets.QLineEdit()
        self.tarot_question.setPlaceholderText("Tarot question...")
        self.btn_tarot = QtWidgets.QPushButton("Tarot Reading")

        btn_row.addWidget(self.intent_edit)
        btn_row.addWidget(self.btn_daily)
        btn_row.addWidget(self.tarot_question)
        btn_row.addWidget(self.btn_tarot)
        layout.addLayout(btn_row)

        self.btn_daily.clicked.connect(self._on_daily)
        self.btn_tarot.clicked.connect(self._on_tarot)

    def _on_daily(self):
        intent = self.intent_edit.text().strip()
        payload = daily_spiritual_message(intent)
        self.universal_chat.append_line(f"SPIRITUAL DAILY: {payload.get('message','')}")

    def _on_tarot(self):
        q = self.tarot_question.text().strip()
        if not q:
            return
        payload = tarot_reading(q)
        self.universal_chat.append_line(f"TAROT: {payload.get('reading','')}")


class PDFCoreTab(QtWidgets.QWidget):
    def __init__(self, universal_chat: UniversalChatManager, parent=None):
        super().__init__(parent)
        self.universal_chat = universal_chat
        self.core = PDFEvolutionCore()
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel("THY KNOWLEDGE")
        header.setAlignment(QtCore.Qt.AlignCenter)
        header.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(header)

        self.chat_box = QtWidgets.QTextEdit()
        self.chat_box.setReadOnly(True)
        self.chat_box.setAcceptDrops(True)
        self.universal_chat.register(self.chat_box)
        layout.addWidget(self.chat_box, 4)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_self_finetune = QtWidgets.QPushButton("SELF Finetune")
        self.btn_summarize = QtWidgets.QPushButton("Summarize")
        self.btn_facts = QtWidgets.QPushButton("Facts")
        self.btn_debate = QtWidgets.QPushButton("Debate")
        self.btn_load = QtWidgets.QPushButton("LOAD")
        self.btn_read_voice = QtWidgets.QPushButton("READ VOICE")
        self.btn_embed = QtWidgets.QPushButton("Embedded")
        for b in (
            self.btn_self_finetune,
            self.btn_summarize,
            self.btn_facts,
            self.btn_debate,
            self.btn_load,
            self.btn_read_voice,
            self.btn_embed,
        ):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        qa_row = QtWidgets.QHBoxLayout()
        self.question_edit = QtWidgets.QLineEdit()
        self.question_edit.setPlaceholderText("Ask a question about the loaded PDF...")
        self.btn_ask = QtWidgets.QPushButton("ASK")
        qa_row.addWidget(self.question_edit)
        qa_row.addWidget(self.btn_ask)
        layout.addLayout(qa_row)

        self.answer_view = QtWidgets.QTextEdit()
        self.answer_view.setReadOnly(True)
        layout.addWidget(self.answer_view, 2)

        self.btn_load.clicked.connect(self._on_load_pdf)
        self.btn_ask.clicked.connect(self._on_ask_question)

    def _on_load_pdf(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select PDF", "", "PDF Files (*.pdf)"
        )
        if not path:
            return
        doc = self.core.load_pdf(Path(path))
        self.chat_box.append(f"Loaded and summarized PDF: {path}")
        self.chat_box.append(doc.summary)
        self.universal_chat.append_line(f"PDF LOADED: {path}")

    def _on_ask_question(self):
        q = self.question_edit.text().strip()
        if not q:
            return
        ans = self.core.ask_question(q)
        self.answer_view.setPlainText(ans)
        self.universal_chat.append_line(f"PDF Q: {q}\nA: {ans}")


class GodAgentsTab(QtWidgets.QWidget):
    def __init__(self, universal_chat: UniversalChatManager, god_agents_core: GodAgentsCore, parent=None):
        super().__init__(parent)
        self.universal_chat = universal_chat
        self.core = god_agents_core
        self._build_ui()

    def _build_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel("THY GOD QUEEN")
        header.setAlignment(QtCore.Qt.AlignCenter)
        header.setStyleSheet("font-size: 18pt; font-weight: bold;")
        main_layout.addWidget(header)

        avatar_row = QtWidgets.QHBoxLayout()
        self.avatar_agents = QtWidgets.QLabel("GOD Agents\n[Animated GIF]")
        self.avatar_agents.setAlignment(QtCore.Qt.AlignCenter)
        self.avatar_agents.setFrameShape(QtWidgets.QFrame.Box)

        self.avatar_queen = QtWidgets.QLabel("GOD QUEEN\n[Animated GIF]")
        self.avatar_queen.setAlignment(QtCore.Qt.AlignCenter)
        self.avatar_queen.setFrameShape(QtWidgets.QFrame.Box)

        avatar_row.addWidget(self.avatar_agents)
        avatar_row.addWidget(self.avatar_queen)
        main_layout.addLayout(avatar_row)

        center_row = QtWidgets.QHBoxLayout()
        self.activity_log = QtWidgets.QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setPlaceholderText("GOD Agents activity log...")
        center_row.addWidget(self.activity_log, 1)

        self.chat_box = QtWidgets.QTextEdit()
        self.chat_box.setReadOnly(True)
        self.chat_box.setPlaceholderText("GOD Agents chatbox...")
        self.universal_chat.register(self.chat_box)
        center_row.addWidget(self.chat_box, 1)
        main_layout.addLayout(center_row)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_enable = QtWidgets.QPushButton("ENABLE GOD QUEEN")
        self.btn_army = QtWidgets.QPushButton("GOD AGENTS ARMY")
        self.btn_scout = QtWidgets.QPushButton("GOD AGENTS SCOUT")
        self.btn_reporter = QtWidgets.QPushButton("GOD AGENTS Reporter")
        self.btn_wizards = QtWidgets.QPushButton("GOD Agents Code WIZARDS")
        self.btn_self_rewrite = QtWidgets.QPushButton("Self-Rewrite")
        self.btn_manual_search = QtWidgets.QPushButton("Manual Search")
        for b in (
            self.btn_enable,
            self.btn_army,
            self.btn_scout,
            self.btn_reporter,
            self.btn_wizards,
            self.btn_self_rewrite,
            self.btn_manual_search,
        ):
            btn_row.addWidget(b)
        main_layout.addLayout(btn_row)

        self.btn_enable.clicked.connect(lambda: self._enqueue(self.core.enable_god_queen, "ENABLE GOD QUEEN"))
        self.btn_army.clicked.connect(lambda: self._enqueue(self.core.run_army_defense, "ARMY"))
        self.btn_scout.clicked.connect(self._on_scout)
        self.btn_reporter.clicked.connect(lambda: self._enqueue(self.core.run_reporter, "Reporter"))
        self.btn_wizards.clicked.connect(lambda: self._enqueue(self.core.run_code_wizards, "Code WIZARDS"))
        self.btn_self_rewrite.clicked.connect(lambda: self._enqueue(self.core.run_self_rewrite, "Self-Rewrite"))
        self.btn_manual_search.clicked.connect(self._on_manual_search)

    def _enqueue(self, fn, label: str):
        self.core.submit_task(fn)
        self.activity_log.append(f"Triggered: {label}")
        self.universal_chat.append_line(f"GOD AGENTS: {label} started.")

    def _on_scout(self):
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self,
            "Scout URLs",
            "Enter one URL per line:",
            "",
        )
        if not ok or not text.strip():
            return
        urls = [line.strip() for line in text.splitlines() if line.strip()]
        self.core.submit_task(self.core.run_scout, urls)
        self.activity_log.append(f"Scout triggered on {len(urls)} URLs.")
        self.universal_chat.append_line("GOD AGENTS: Scout activated.")

    def _on_manual_search(self):
        q, ok = QtWidgets.QInputDialog.getText(
            self,
            "Manual Search",
            "Enter search query:",
        )
        if not ok or not q.strip():
            return
        self.core.submit_task(self.core.manual_search, q)
        self.activity_log.append(f"Manual search for: {q}")
        self.universal_chat.append_line(f"GOD AGENTS: manual search = {q}")


class MemoryCoreTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.memory_core = MemoryCore()
        self._build_ui()
        self.refresh_memory_stats()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel("THY MEMORY")
        header.setAlignment(QtCore.Qt.AlignCenter)
        header.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(header)

        graph_row = QtWidgets.QHBoxLayout()
        self.short_term_bar = QtWidgets.QProgressBar()
        self.short_term_bar.setFormat("Short Term Memory: %p%")
        self.long_term_bar = QtWidgets.QProgressBar()
        self.long_term_bar.setFormat("Long Term Memory: %p%")
        graph_row.addWidget(self.short_term_bar)
        graph_row.addWidget(self.long_term_bar)
        layout.addLayout(graph_row)

        btn_row1 = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("REFRESH MEMORY")
        self.btn_cleanup = QtWidgets.QPushButton("Clean UP Memory")
        self.btn_optimize = QtWidgets.QPushButton("Optimize Memory")
        btn_row1.addWidget(self.btn_refresh)
        btn_row1.addWidget(self.btn_cleanup)
        btn_row1.addWidget(self.btn_optimize)
        layout.addLayout(btn_row1)

        btn_row2 = QtWidgets.QHBoxLayout()
        self.btn_opt_kb = QtWidgets.QPushButton("Optimize Knowledgebase")
        self.btn_load_ds = QtWidgets.QPushButton("Load Dataset")
        self.btn_self_finetune = QtWidgets.QPushButton("SELF FINETUNE")
        btn_row2.addWidget(self.btn_opt_kb)
        btn_row2.addWidget(self.btn_load_ds)
        btn_row2.addWidget(self.btn_self_finetune)
        layout.addLayout(btn_row2)

        storage_row = QtWidgets.QHBoxLayout()
        self.memory_storage_bar = QtWidgets.QProgressBar()
        self.memory_storage_bar.setFormat("MEMORY STORAGE: %p%")
        self.db_storage_bar = QtWidgets.QProgressBar()
        self.db_storage_bar.setFormat("DATABASE STORAGE: %p%")
        storage_row.addWidget(self.memory_storage_bar)
        storage_row.addWidget(self.db_storage_bar)
        layout.addLayout(storage_row)

        logs_row = QtWidgets.QHBoxLayout()
        self.activity_log = QtWidgets.QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setPlaceholderText("Activity LOG")

        self.error_log = QtWidgets.QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setPlaceholderText("ERROR LOGS")

        self.diary_log = QtWidgets.QTextEdit()
        self.diary_log.setPlaceholderText("Personal Diary")

        self.goals_log = QtWidgets.QTextEdit()
        self.goals_log.setPlaceholderText("Future Goals")

        logs_row.addWidget(self.activity_log)
        logs_row.addWidget(self.error_log)
        logs_row.addWidget(self.diary_log)
        logs_row.addWidget(self.goals_log)
        layout.addLayout(logs_row)

        self.btn_refresh.clicked.connect(self.refresh_memory_stats)
        self.btn_cleanup.clicked.connect(self.cleanup_memory)
        self.btn_optimize.clicked.connect(self.optimize_memory)
        self.btn_opt_kb.clicked.connect(self.optimize_knowledgebase)
        self.btn_load_ds.clicked.connect(self.load_dataset)
        self.btn_self_finetune.clicked.connect(self.self_finetune)

    def refresh_memory_stats(self):
        recent_conv = self.memory_core.fetch_recent("conversation", limit=1000)
        recent_pdf = self.memory_core.fetch_recent("pdf_summary", limit=1000)
        total_short = len(recent_conv)
        total_long = len(recent_pdf)

        self.short_term_bar.setMaximum(1000)
        self.short_term_bar.setValue(min(total_short, 1000))

        self.long_term_bar.setMaximum(1000)
        self.long_term_bar.setValue(min(total_long, 1000))

    def cleanup_memory(self):
        conn = self.memory_core._conn_handle
        cutoff = time.time() - 90 * 86400
        conn.execute("DELETE FROM memory WHERE created < ?", (cutoff,))
        conn.commit()
        self.activity_log.append("Cleanup: removed records older than 90 days.")
        self.refresh_memory_stats()

    def optimize_memory(self):
        self.memory_core.vacuum()
        self.activity_log.append("Optimize: VACUUM completed.")

    def optimize_knowledgebase(self):
        self.activity_log.append("Optimize Knowledgebase: extend as needed.")

    def load_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Dataset (text)", "", "Text Files (*.txt);;All Files (*.*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load dataset:\n{e}")
            return
        rec = MemoryRecord(
            kind="dataset",
            content=content,
            metadata={"path": path},
        )
        self.memory_core.store(rec)
        self.activity_log.append(f"Loaded dataset into memory: {path}")
        self.refresh_memory_stats()

    def self_finetune(self):
        note = "SELF FINETUNE requested via MemoryCoreTab."
        rec = MemoryRecord(
            kind="self_finetune_request",
            content=note,
            metadata={},
        )
        self.memory_core.store(rec)
        self.activity_log.append(note)


class SettingsTab(QtWidgets.QWidget):
    settings_changed = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings: Dict[str, Any] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel("Settings & Options")
        header.setAlignment(QtCore.Qt.AlignCenter)
        header.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(header)

        form = QtWidgets.QFormLayout()

        self.font_size_spin = QtWidgets.QSpinBox()
        self.font_size_spin.setRange(8, 32)
        self.font_size_spin.setValue(11)
        form.addRow("GUI Font Size", self.font_size_spin)

        self.color_theme_combo = QtWidgets.QComboBox()
        self.color_theme_combo.addItems(["System", "Light", "Dark"])
        form.addRow("GUI Theme", self.color_theme_combo)

        self.window_size_combo = QtWidgets.QComboBox()
        self.window_size_combo.addItems(["800x600", "1000x700", "1280x800", "1920x1080"])
        form.addRow("GUI Size", self.window_size_combo)

        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItems(["English", "Spanish", "French"])
        form.addRow("Language", self.language_combo)

        self.system_options_edit = QtWidgets.QLineEdit()
        self.system_options_edit.setPlaceholderText("e.g. CPU-only, disable GPU offload")
        form.addRow("CPU/GPU/RAM Options", self.system_options_edit)

        self.voice_options_edit = QtWidgets.QLineEdit()
        self.voice_options_edit.setPlaceholderText("Voice model / microphone notes")
        form.addRow("Voice Settings", self.voice_options_edit)

        self.folder_paths_edit = QtWidgets.QLineEdit()
        self.folder_paths_edit.setPlaceholderText("Override base folder paths if needed")
        form.addRow("File Folder Paths", self.folder_paths_edit)

        self.nsfw_checkbox = QtWidgets.QCheckBox("Enable NSFW")
        form.addRow("NSFW", self.nsfw_checkbox)

        self.server_options_edit = QtWidgets.QLineEdit()
        self.server_options_edit.setPlaceholderText("Server host:port or notes")
        form.addRow("Server Options", self.server_options_edit)

        layout.addLayout(form)

        self.btn_apply = QtWidgets.QPushButton("Apply Settings")
        layout.addWidget(self.btn_apply)

        self.btn_apply.clicked.connect(self._on_apply)

    def _on_apply(self):
        self._settings = {
            "font_size": self.font_size_spin.value(),
            "theme": self.color_theme_combo.currentText(),
            "window_size": self.window_size_combo.currentText(),
            "language": self.language_combo.currentText(),
            "system_options": self.system_options_edit.text(),
            "voice_options": self.voice_options_edit.text(),
            "folder_paths": self.folder_paths_edit.text(),
            "nsfw_enabled": self.nsfw_checkbox.isChecked(),
            "server_options": self.server_options_edit.text(),
        }
        self.settings_changed.emit(self._settings)


class ResourceMetersWidget(QtWidgets.QWidget):
    def __init__(self, god_agents_core: GodAgentsCore, parent=None):
        super().__init__(parent)
        self.god_agents_core = god_agents_core
        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.gpu_bar = QtWidgets.QProgressBar()
        self.gpu_bar.setFormat("GPU VRAM: %p%")
        self.cpu_bar = QtWidgets.QProgressBar()
        self.cpu_bar.setFormat("CPU: %p%")
        self.ram_bar = QtWidgets.QProgressBar()
        self.ram_bar.setFormat("RAM: %p%")
        self.god_agents_label = QtWidgets.QLabel("GOD Agents Active: NO")

        layout.addWidget(self.gpu_bar)
        layout.addWidget(self.cpu_bar)
        layout.addWidget(self.ram_bar)
        layout.addWidget(self.god_agents_label)

    def update_metrics(self):
        try:
            import psutil
        except Exception:
            self.cpu_bar.setValue(0)
            self.ram_bar.setValue(0)
        else:
            self.cpu_bar.setValue(int(psutil.cpu_percent()))
            mem = psutil.virtual_memory()
            self.ram_bar.setValue(int(mem.percent))

        self.gpu_bar.setValue(0)

        active = any(
            getattr(self.god_agents_core, name, False)
            for name in (
                "army_active",
                "scout_active",
                "reporter_active",
                "code_wizards_active",
                "self_rewrite_active",
            )
        )
        self.god_agents_label.setText(f"GOD Agents Active: {'YES' if active else 'NO'}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hope DuGan AI")

        self.universal_chat = UniversalChatManager()
        self.god_agents_core = GodAgentsCore()

        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)

        sidebar = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)

        self.btn_chat = QtWidgets.QPushButton("CHAT CORE")
        self.btn_gen = QtWidgets.QPushButton("GENERATIONAL CORE")
        self.btn_spiritual = QtWidgets.QPushButton("Spiritual Core")
        self.btn_pdf = QtWidgets.QPushButton("PDF/Knowledge CORE")
        self.btn_god = QtWidgets.QPushButton("GOD Agents Core")
        self.btn_memory = QtWidgets.QPushButton("MEMORY CORE")
        self.btn_settings = QtWidgets.QPushButton("Settings")

        for b in (
            self.btn_chat,
            self.btn_gen,
            self.btn_spiritual,
            self.btn_pdf,
            self.btn_god,
            self.btn_memory,
            self.btn_settings,
        ):
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            sidebar_layout.addWidget(b)

        sidebar_layout.addStretch(1)

        self.resource_meters = ResourceMetersWidget(self.god_agents_core)
        sidebar_layout.addWidget(self.resource_meters)

        main_layout.addWidget(sidebar, 1)

        self.stack = QtWidgets.QStackedWidget()

        self.chat_tab = ChatCoreTab(self.universal_chat)
        self.gen_tab = GenerationalCoreTab(self.universal_chat)
        self.spiritual_tab = SpiritualCoreTab(self.universal_chat)
        self.pdf_tab = PDFCoreTab(self.universal_chat)
        self.god_tab = GodAgentsTab(self.universal_chat, self.god_agents_core)
        self.memory_tab = MemoryCoreTab()
        self.settings_tab = SettingsTab()

        self.stack.addWidget(self.chat_tab)
        self.stack.addWidget(self.gen_tab)
        self.stack.addWidget(self.spiritual_tab)
        self.stack.addWidget(self.pdf_tab)
        self.stack.addWidget(self.god_tab)
        self.stack.addWidget(self.memory_tab)
        self.stack.addWidget(self.settings_tab)

        main_layout.addWidget(self.stack, 4)

        self.btn_chat.clicked.connect(lambda: self.stack.setCurrentWidget(self.chat_tab))
        self.btn_gen.clicked.connect(lambda: self.stack.setCurrentWidget(self.gen_tab))
        self.btn_spiritual.clicked.connect(lambda: self.stack.setCurrentWidget(self.spiritual_tab))
        self.btn_pdf.clicked.connect(lambda: self.stack.setCurrentWidget(self.pdf_tab))
        self.btn_god.clicked.connect(lambda: self.stack.setCurrentWidget(self.god_tab))
        self.btn_memory.clicked.connect(lambda: self.stack.setCurrentWidget(self.memory_tab))
        self.btn_settings.clicked.connect(lambda: self.stack.setCurrentWidget(self.settings_tab))

        self.settings_tab.settings_changed.connect(self.apply_settings)

        self.setCentralWidget(central)

    def apply_settings(self, settings: Dict[str, Any]):
        font = self.font()
        font.setPointSize(settings.get("font_size", 11))
        self.setFont(font)

        theme = settings.get("theme", "System")
        if theme == "Dark":
            self.setStyleSheet("QWidget { background-color: #202020; color: #f0f0f0; }")
        elif theme == "Light":
            self.setStyleSheet("")
        else:
            self.setStyleSheet("")

        size_map = {
            "800x600": (800, 600),
            "1000x700": (1000, 700),
            "1280x800": (1280, 800),
            "1920x1080": (1920, 1080),
        }
        size_str = settings.get("window_size", "1000x700")
        if size_str in size_map:
            w, h = size_map[size_str]
            self.resize(w, h)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1000, 700)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()