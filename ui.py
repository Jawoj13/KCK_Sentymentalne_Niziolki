__all__ = ['MainWindow']

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QWidget, QLabel, QHBoxLayout, QVBoxLayout,
                             QLineEdit, QPushButton, QSizePolicy, QGroupBox,
                             QTextEdit, QStackedWidget)


class MainWindow(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Trener Jugger")
		self.resize(1280, 720)  # rozmiar okna startowego

		# --- WIDŻET STOSOWY (QStackedWidget) ---
		self.stacked_widget = QStackedWidget()

		# Inicjalizacja poszczególnych ekranów aplikacji
		self.init_menu_page()  # Ekran 0: Menu Główne
		self.init_training_page()  # Ekran 1: Kamery i Trening
		self.init_settings_page()  # Ekran 2: Ustawienia

		# Dodanie ekranów do kontenera (indeksy: 0, 1, 2)
		self.stacked_widget.addWidget(self.menu_page)
		self.stacked_widget.addWidget(self.training_page)
		self.stacked_widget.addWidget(self.settings_page)

		# Główny układ okna
		main_layout = QVBoxLayout()
		main_layout.addWidget(self.stacked_widget)
		self.setLayout(main_layout)

		# Ekran startowy (Menu Główne)
		self.stacked_widget.setCurrentIndex(0)

	# EKRAN 0: MENU GŁÓWNE
	def init_menu_page(self):
		self.menu_page = QWidget()
		menu_layout = QVBoxLayout()
		menu_layout.setAlignment(Qt.AlignCenter)

		title = QLabel("TRENER JUGGER")
		title.setAlignment(Qt.AlignCenter)
		title.setStyleSheet("font-size: 36px; font-weight: bold; margin-bottom: 40px; color: #2c3e50;")
		menu_layout.addWidget(title)

		self.btn_train = QPushButton("Trenuj")
		self.btn_settings = QPushButton("Ustawienia")
		self.btn_exit = QPushButton("Wyjdź")

		button_style = "font-size: 18px; padding: 15px; min-width: 250px; margin: 8px;"
		for btn in (self.btn_train, self.btn_settings, self.btn_exit):
			btn.setStyleSheet(button_style)
			menu_layout.addWidget(btn)

		self.btn_train.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
		self.btn_settings.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
		self.btn_exit.clicked.connect(self.close)

		self.menu_page.setLayout(menu_layout)

	# EKRAN 1: PANEL TRENINGU (KAMERY)
	def init_training_page(self):
		self.training_page = QWidget()
		training_layout = QHBoxLayout()

		# --- LEWA STRONA: WIDOKI KAMER ---
		video_area_layout = QVBoxLayout()
		cameras_layout = QHBoxLayout()

		self.label_a = QLabel("Kamera: Laptop")
		self.label_b = QLabel("Kamera: Telefon (Oczekiwanie...)")

		self.video_preview_width = 560
		self.video_preview_height = 420

		for label in (self.label_a, self.label_b):
			label.setAlignment(Qt.AlignCenter)
			label.setFixedSize(self.video_preview_width, self.video_preview_height)
			label.setStyleSheet("background-color: #000; color: #fff; border: 1px solid #555;")
			cameras_layout.addWidget(label)

		video_area_layout.addLayout(cameras_layout)

		self.status_banner = QLabel("STATUS: Gotowy do treningu")
		self.status_banner.setAlignment(Qt.AlignCenter)
		self.status_banner.setWordWrap(True)
		self.status_banner.setMinimumHeight(70)
		self.status_banner.setStyleSheet(
			"font-size: 20px; font-weight: bold; padding: 10px; background-color: #2ecc71; color: black;")
		video_area_layout.addWidget(self.status_banner)

		# --- PRAWA STRONA: PANEL KOMUNIKATÓW ---
		control_panel_layout = QVBoxLayout()
		control_panel_layout.setAlignment(Qt.AlignTop)

		logs_group = QGroupBox("Komunikaty systemu (Asystent)")
		logs_vbox = QVBoxLayout()
		self.log_console = QTextEdit()
		self.log_console.setReadOnly(True)
		self.log_console.setPlaceholderText(
			"Tutaj pojawią się alerty o błędach w postawie lub zbyt obszernych zamachach...")
		logs_vbox.addWidget(self.log_console)
		logs_group.setLayout(logs_vbox)
		control_panel_layout.addWidget(logs_group)

		# Przycisk powrotu do menu głównego
		self.btn_back_from_train = QPushButton("← Powrót do Menu")
		self.btn_back_from_train.setStyleSheet(
			"padding: 10px; background-color: #e74c3c; color: white; font-weight: bold;")
		self.btn_back_from_train.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
		control_panel_layout.addWidget(self.btn_back_from_train)

		training_layout.addLayout(video_area_layout, stretch=3)
		training_layout.addLayout(control_panel_layout, stretch=1)

		self.training_page.setLayout(training_layout)

	# EKRAN 2: USTAWIENIA (POŁĄCZENIE IP + KOD QR / LINK)

	def init_settings_page(self):
		import os

		self.settings_page = QWidget()
		settings_layout = QVBoxLayout()
		settings_layout.setAlignment(Qt.AlignTop)

		title = QLabel("Ustawienia Aplikacji")
		title.setAlignment(Qt.AlignCenter)
		title.setStyleSheet(
			"font-size: 24px; font-weight: bold; margin-top: 10px; margin-bottom: 25px; color: #2c3e50;")
		settings_layout.addWidget(title)

		# SEKCJA KONFIGURACJI KAMERY IP
		conn_group = QGroupBox("Połączenie z drugą kamerą (Telefon)")
		conn_vbox = QVBoxLayout()

		self.ip_input = QLineEdit("http://192.168.1.14:8080/video")
		self.ip_input.setPlaceholderText("Wpisz adres URL z aplikacji IP Webcam...")
		self.ip_input.setStyleSheet("padding: 8px; font-size: 14px;")

		self.connect_btn = QPushButton("Połącz i zapamiętaj kamerę")
		self.connect_btn.setStyleSheet("padding: 10px; font-weight: bold; background-color: #3498db; color: white;")

		conn_vbox.addWidget(self.ip_input)
		conn_vbox.addWidget(self.connect_btn)

		# --- DYNAMICZNE SZUKANIE KODU QR ---
		self.qr_label = QLabel()
		self.qr_label.setAlignment(Qt.AlignCenter)

		sciezka_folderu = os.path.dirname(os.path.abspath(__file__))
		pelna_sciezka_qr = os.path.join(sciezka_folderu, "qr_kod.png")

		qr_pixmap = QPixmap(pelna_sciezka_qr)

		if not qr_pixmap.isNull():
			# Jeśli obrazek istnieje pod wskazaną pełną ścieżką
			qr_pixmap = qr_pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
			self.qr_label.setPixmap(qr_pixmap)
			self.qr_label.setStyleSheet("margin-top: 15px;")

			qr_info = QLabel("Zeskanuj, aby pobrać aplikację IP Webcam z Google Play")
			qr_info.setAlignment(Qt.AlignCenter)
			qr_info.setStyleSheet("color: #7f8c8d; font-size: 12px; margin-bottom: 10px;")

			conn_vbox.addWidget(self.qr_label)
			conn_vbox.addWidget(qr_info)
		else:
			# Jeśli z jakiegoś powodu pliku wciąż nie ma, dajemy klikalny link
			link_text = '<a href="https://play.google.com/store/apps/details?id=com.pas.webcam" style="color: #3498db; text-decoration: none;">Brak pliku kodu QR.<br><b>Kliknij tutaj</b>, aby pobrać IP Webcam z Google Play</a>'
			self.qr_label.setText(link_text)
			self.qr_label.setOpenExternalLinks(True)
			self.qr_label.setStyleSheet("font-size: 14px; margin-top: 20px; margin-bottom: 20px;")
			conn_vbox.addWidget(self.qr_label)

		conn_group.setLayout(conn_vbox)
		settings_layout.addWidget(conn_group)

		# Sekcja na przyszłe parametry YOLO
		yolo_group = QGroupBox("Zaawansowane parametry YOLOv8")
		yolo_vbox = QVBoxLayout()
		info = QLabel("Miejsce na przyszłą konfigurację algorytmów (np. progi detekcji, czułość wyłapywania zamachów).")
		info.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 5px;")
		yolo_vbox.addWidget(info)
		yolo_group.setLayout(yolo_vbox)
		settings_layout.addWidget(yolo_group)

		settings_layout.addStretch()

		# Przycisk powrotu do menu głównego
		self.btn_back_from_settings = QPushButton("← Powrót do Menu")
		self.btn_back_from_settings.setStyleSheet("font-size: 16px; padding: 10px;")
		self.btn_back_from_settings.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
		settings_layout.addWidget(self.btn_back_from_settings)

		self.settings_page.setLayout(settings_layout)

	# METODA REFRESHUJĄCA OBRAZ
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
				Qt.KeepAspectRatio,
				Qt.SmoothTransformation,
			)
			self.label_b.setPixmap(scaled_b)

	def append_evaluation_result(self, result):
		if not result:
			return

		score = result.get("score", "N/A")
		confidence = result.get("confidence", "N/A")
		main_feedback = result.get("main_feedback") or result.get("feedback")
		component_scores = result.get("component_scores", {})

		if isinstance(main_feedback, dict):
			feedback_message = main_feedback.get("message", "Brak informacji zwrotnej")
			feedback_code = main_feedback.get("code", "UNKNOWN")
		else:
			feedback_message = main_feedback or "Brak informacji zwrotnej"
			feedback_code = None

		message_lines = [
			f"Score: {score}",
			f"Confidence: {confidence}",
			f"Main feedback: {feedback_message}",
		]

		if feedback_code:
			message_lines.append(f"Feedback code: {feedback_code}")

		message_lines.append("Component scores:")

	def update_debug_status(self, text):
		self.status_banner.setText(text)
