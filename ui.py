__all__ = ['MainWindow']

import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QLabel, QHBoxLayout, QVBoxLayout,
                             QLineEdit, QPushButton, QGroupBox,
                             QTextEdit, QStackedWidget)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trener Jugger")
        self.resize(1920, 1080)
        self.setMinimumSize(1280, 720)

        self.stacked_widget = QStackedWidget()

        self.init_menu_page()
        self.init_training_page()
        self.init_settings_page()
        self.init_history_page()

        self.stacked_widget.addWidget(self.menu_page)
        self.stacked_widget.addWidget(self.training_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.history_page)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)

        self.stacked_widget.setCurrentIndex(0)

    def init_menu_page(self):
        self.menu_page = QWidget()
        menu_layout = QVBoxLayout()
        menu_layout.setAlignment(Qt.AlignCenter)

        title = QLabel("TRENER JUGGER")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 36px; font-weight: bold; margin-bottom: 40px; color: #2c3e50;")
        menu_layout.addWidget(title)

        self.btn_train = QPushButton("Trenuj")
        self.btn_history = QPushButton("Historia ćwiczeń")
        self.btn_settings = QPushButton("Ustawienia")
        self.btn_exit = QPushButton("Wyjdź")

        button_style = "font-size: 18px; padding: 15px; min-width: 250px; margin: 8px;"
        for btn in (self.btn_train, self.btn_history, self.btn_settings, self.btn_exit):
            btn.setStyleSheet(button_style)
            menu_layout.addWidget(btn)

        self.btn_train.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.btn_history.clicked.connect(self.open_history)
        self.btn_settings.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        self.btn_exit.clicked.connect(self.close)

        self.menu_page.setLayout(menu_layout)

    def init_training_page(self):
        self.training_page = QWidget()
        training_layout = QHBoxLayout()

        video_area_layout = QVBoxLayout()
        cameras_layout = QHBoxLayout()

        self.label_a = QLabel("Kamera: Laptop")
        self.label_b = QLabel("Kamera: Telefon (Oczekiwanie...)")

        self.video_preview_width = 480
        self.video_preview_height = 360

        for label in (self.label_a, self.label_b):
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(self.video_preview_width, self.video_preview_height)
            label.setStyleSheet("background-color: #000; color: #fff; border: 1px solid #555;")
            cameras_layout.addWidget(label)

        video_area_layout.addLayout(cameras_layout)

        self.status_banner = QLabel("STATUS: Gotowy do treningu")
        self.status_banner.setAlignment(Qt.AlignCenter)
        self.status_banner.setFixedHeight(48)
        self.status_banner.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 8px; background-color: #2ecc71; color: black;"
        )
        video_area_layout.addWidget(self.status_banner)

        control_panel_layout = QVBoxLayout()
        control_panel_layout.setAlignment(Qt.AlignTop)

        debug_group = QGroupBox("Debug segmentera")
        debug_group.setMinimumHeight(335)
        debug_group.setMaximumHeight(360)

        debug_vbox = QVBoxLayout()
        debug_vbox.setContentsMargins(8, 18, 8, 8)

        self.debug_console = QTextEdit()
        self.debug_console.setReadOnly(True)
        self.debug_console.setFixedHeight(285)
        self.debug_console.setLineWrapMode(QTextEdit.NoWrap)
        self.debug_console.setStyleSheet(
            "font-family: monospace; font-size: 12px; background-color: #111; color: #00ff88;"
        )

        debug_vbox.addWidget(self.debug_console)
        debug_group.setLayout(debug_vbox)

        control_panel_layout.addWidget(debug_group)
        result_group = QGroupBox("Szczegóły bieżącego powtórzenia")
        result_vbox = QVBoxLayout()

        self.result_details_console = QTextEdit()
        self.result_details_console.setReadOnly(True)
        self.result_details_console.setMinimumHeight(210)
        self.result_details_console.setStyleSheet(
            "font-family: monospace; font-size: 12px; background-color: #1f2933; color: #f5f5f5;"
        )

        result_vbox.addWidget(self.result_details_console)
        result_group.setLayout(result_vbox)
        control_panel_layout.addWidget(result_group)

        logs_group = QGroupBox("Komunikaty systemu (Asystent - Podsumowania Serii)")
        logs_vbox = QVBoxLayout()
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setPlaceholderText("Tutaj pojawią się zagregowane podsumowania serii powtórzeń...")
        self.log_console.setStyleSheet("font-size: 14px;")
        logs_vbox.addWidget(self.log_console)
        logs_group.setLayout(logs_vbox)
        control_panel_layout.addWidget(logs_group)

        self.btn_back_from_train = QPushButton("← Powrót do Menu")
        self.btn_back_from_train.setStyleSheet(
            "padding: 10px; background-color: #e74c3c; color: white; font-weight: bold;")
        self.btn_back_from_train.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        control_panel_layout.addWidget(self.btn_back_from_train)

        control_panel_widget = QWidget()
        control_panel_widget.setMinimumWidth(500)
        control_panel_widget.setMaximumWidth(620)
        control_panel_widget.setLayout(control_panel_layout)

        training_layout.addLayout(video_area_layout, stretch=1)
        training_layout.addWidget(control_panel_widget)

        self.training_page.setLayout(training_layout)

    def init_settings_page(self):
        self.settings_page = QWidget()
        settings_layout = QVBoxLayout()
        settings_layout.setAlignment(Qt.AlignTop)

        title = QLabel("Ustawienia Aplikacji")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold; margin-top: 10px; margin-bottom: 25px; color: #2c3e50;")
        settings_layout.addWidget(title)

        # SEKCJA KAMERY
        conn_group = QGroupBox("Połączenie z drugą kamerą (Telefon)")
        conn_vbox = QVBoxLayout()

        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Wpisz adres URL z aplikacji IP Webcam...")
        self.ip_input.setStyleSheet("padding: 8px; font-size: 14px;")

        self.connect_btn = QPushButton("Połącz z kamerą")
        self.connect_btn.setStyleSheet("padding: 10px; font-weight: bold; background-color: #3498db; color: white;")

        conn_vbox.addWidget(self.ip_input)
        conn_vbox.addWidget(self.connect_btn)

        self.qr_label = QLabel()
        self.qr_label.setAlignment(Qt.AlignCenter)
        sciezka_folderu = os.path.dirname(os.path.abspath(__file__))
        pelna_sciezka_qr = os.path.join(sciezka_folderu, "qr_kod.png")
        qr_pixmap = QPixmap(pelna_sciezka_qr)

        if not qr_pixmap.isNull():
            qr_pixmap = qr_pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.qr_label.setPixmap(qr_pixmap)
            self.qr_label.setStyleSheet("margin-top: 15px;")
            qr_info = QLabel("Zeskanuj, aby pobrać aplikację IP Webcam z Google Play")
            qr_info.setAlignment(Qt.AlignCenter)
            qr_info.setStyleSheet("color: #7f8c8d; font-size: 12px; margin-bottom: 10px;")
            conn_vbox.addWidget(self.qr_label)
            conn_vbox.addWidget(qr_info)
        else:
            link_text = '<a href="https://play.google.com/store/apps/details?id=com.pas.webcam" style="color: #3498db; text-decoration: none;">Brak pliku kodu QR.<br><b>Kliknij tutaj</b>, aby pobrać IP Webcam z Google Play</a>'
            self.qr_label.setText(link_text)
            self.qr_label.setOpenExternalLinks(True)
            self.qr_label.setStyleSheet("font-size: 14px; margin-top: 20px; margin-bottom: 20px;")
            conn_vbox.addWidget(self.qr_label)

        conn_group.setLayout(conn_vbox)
        settings_layout.addWidget(conn_group)

        # SEKCJA LOGÓW I PODSUMOWAŃ
        log_group = QGroupBox("Historia ćwiczeń i Podsumowania (log.txt)")
        log_vbox = QVBoxLayout()

        self.summary_input = QLineEdit()
        self.summary_input.setPlaceholderText("Liczba powtórzeń do wygenerowania podsumowania (np. 5)")
        self.summary_input.setStyleSheet("padding: 8px; font-size: 14px;")

        self.retention_input = QLineEdit()
        self.retention_input.setPlaceholderText("Czas przechowywania logów w dniach (np. 30)")
        self.retention_input.setStyleSheet("padding: 8px; font-size: 14px;")

        self.save_settings_btn = QPushButton("Zapisz ustawienia ręcznie")
        self.save_settings_btn.setStyleSheet(
            "padding: 10px; font-weight: bold; background-color: #2ecc71; color: white;")

        log_vbox.addWidget(QLabel("Liczba powtórzeń tworzących jedną serię/podsumowanie:"))
        log_vbox.addWidget(self.summary_input)
        log_vbox.addWidget(QLabel("Czas przechowywania logów (w dniach):"))
        log_vbox.addWidget(self.retention_input)
        log_vbox.addWidget(self.save_settings_btn)

        log_group.setLayout(log_vbox)
        settings_layout.addWidget(log_group)

        settings_layout.addStretch()

        self.btn_back_from_settings = QPushButton("← Powrót do Menu")
        self.btn_back_from_settings.setStyleSheet("font-size: 16px; padding: 10px;")
        self.btn_back_from_settings.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        settings_layout.addWidget(self.btn_back_from_settings)

        self.settings_page.setLayout(settings_layout)

    def init_history_page(self):
        self.history_page = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Historia Treningów (Zestawienia Serii)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold; margin-top: 10px; margin-bottom: 15px; color: #2c3e50;")
        layout.addWidget(title)

        self.history_console = QTextEdit()
        self.history_console.setReadOnly(True)
        self.history_console.setStyleSheet(
            "font-family: monospace; font-size: 14px; background-color: #1f2933; color: #f5f5f5; padding: 10px;"
        )
        layout.addWidget(self.history_console)

        self.btn_back_from_history = QPushButton("← Powrót do Menu")
        self.btn_back_from_history.setStyleSheet("font-size: 16px; padding: 10px; margin-top: 10px;")
        self.btn_back_from_history.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        layout.addWidget(self.btn_back_from_history)

        self.history_page.setLayout(layout)

    def open_history(self):
        if os.path.exists("log.txt"):
            try:
                with open("log.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        self.history_console.setPlainText(content)
                        scrollbar = self.history_console.verticalScrollBar()
                        scrollbar.setValue(scrollbar.maximum())
                    else:
                        self.history_console.setPlainText("Brak historii ćwiczeń (plik jest pusty).")
            except Exception as e:
                self.history_console.setPlainText(f"Błąd podczas ładowania pliku log.txt:\n{e}")
        else:
            self.history_console.setPlainText("Brak historii ćwiczeń (plik log.txt jeszcze nie istnieje).")

        self.stacked_widget.setCurrentIndex(3)

    def update_both_labels(self, q_img_a, q_img_b):
        if not q_img_a.isNull():
            pixmap_a = QPixmap.fromImage(q_img_a)
            scaled_a = pixmap_a.scaled(
                self.video_preview_width,
                self.video_preview_height,
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            self.label_a.setPixmap(scaled_a)

        if not q_img_b.isNull():
            pixmap_b = QPixmap.fromImage(q_img_b)
            scaled_b = pixmap_b.scaled(
                self.video_preview_width,
                self.video_preview_height,
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            self.label_b.setPixmap(scaled_b)

    def update_debug_status(self, text):
        self.status_banner.setText("STATUS: Analiza aktywna")
        self.debug_console.setPlainText(text)

    def format_repetition_details(self, result):
        if not result:
            return "Brak wyniku."

        exercise_type = result.get("exercise_type", "unknown")
        score = result.get("score", "N/A")
        main_feedback = result.get("main_feedback")
        errors = result.get("errors", [])

        lines = [
            "BIEŻĄCE POWTÓRZENIE",
            f"Typ: {exercise_type} | Wynik: {score}",
            "",
            "BŁĘDY W TYM RUCHU:"
        ]

        if errors:
            for e in errors:
                lines.append(f"- {e.get('message', 'Brak komunikatu')}")
        else:
            lines.append("- Brak (ruch poprawny!)")

        return "\n".join(lines)