import cv2
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trener Jugger")
        self.resize(1920, 1080)

        # Pole tekstowe i przycisk
        self.ip_input = QLineEdit("http://192.168.1.14:8080/video")
        self.ip_input.setPlaceholderText("Paste IP Webcam URL here...")
        self.connect_btn = QPushButton("Connect Camera")

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.ip_input)
        input_layout.addWidget(self.connect_btn)

        # Ekrany kamer
        self.label_a = QLabel("Laptop...")
        self.label_b = QLabel("Waiting for IP Connection...")
        self.label_a.setAlignment(Qt.AlignCenter)
        self.label_b.setAlignment(Qt.AlignCenter)

        # Lock layout to prevent flickering
        self.label_a.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_a.setMinimumSize(640, 480)
        self.label_b.setMinimumSize(640, 480)

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.label_a)
        video_layout.addWidget(self.label_b)

        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(video_layout)

        self.setLayout(main_layout)

    # Główna metoda aktualizująca interfejs po otrzymaniu nowych klatek z SyncInferenceWorker.
    def update_both_labels(self, q_img_a, q_img_b):
        if not q_img_a.isNull() and self.label_a.width() > 0 and self.label_a.height() > 0:
            pixmap_a = QPixmap.fromImage(q_img_a)
            scaled_a = pixmap_a.scaled(self.label_a.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if not scaled_a.isNull():
                self.label_a.setPixmap(scaled_a)

        if not q_img_b.isNull() and self.label_b.width() > 0 and self.label_b.height() > 0:
            pixmap_b = QPixmap.fromImage(q_img_b)
            scaled_b = pixmap_b.scaled(self.label_b.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if not scaled_b.isNull():
                self.label_b.setPixmap(scaled_b)