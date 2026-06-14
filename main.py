import json
import os
import queue
import sys
from collections import Counter
from datetime import datetime, timedelta

from PyQt5.QtWidgets import QApplication

from backend import CameraWorker, SyncInferenceWorker
from ui import MainWindow

# Konfiguracja nazw plików
SETTINGS_FILE = "settings.txt"
LOG_FILE = "log.txt"


def load_settings():
    default_settings = {
        "ip_camera": "http://192.168.1.14:8080/video",
        "log_retention_days": 30,
        "summary_frequency": 5,
        "exercise_type": "full"  # Zapewnienie domyślnej wartości dla ćwiczenia
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                default_settings.update(loaded)
        except Exception as e:
            print(f"Błąd ładowania ustawień: {e}")
    return default_settings


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"Błąd zapisywania ustawień: {e}")


def clean_old_logs(retention_days):
    if not os.path.exists(LOG_FILE):
        return

    cutoff_date = datetime.now() - timedelta(days=retention_days)

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return

        entries = content.split("=== [")
        valid_entries = []

        for entry in entries:
            if not entry.strip():
                continue
            try:
                date_str = entry[:19]
                entry_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                if entry_date >= cutoff_date:
                    valid_entries.append("=== [" + entry)
            except ValueError:
                valid_entries.append("=== [" + entry)

        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("".join(valid_entries))

    except Exception as e:
        print(f"Błąd podczas czyszczenia logów: {e}")


class AppController:
    def __init__(self):
        self.window = MainWindow()

        # --- INICJALIZACJA USTAWIEŃ ---
        self.settings = load_settings()
        retention_days = self.settings.get("log_retention_days", 30)
        self.summary_frequency = self.settings.get("summary_frequency", 5)
        exercise_type = self.settings.get("exercise_type", "full")

        clean_old_logs(retention_days)

        # Wypełnienie pól UI
        self.window.ip_input.setText(self.settings.get("ip_camera", ""))
        if hasattr(self.window, "retention_input"):
            self.window.retention_input.setText(str(retention_days))
        if hasattr(self.window, "summary_input"):
            self.window.summary_input.setText(str(self.summary_frequency))

        # Ustawienie wybranego ćwiczenia z pliku settings
        index = self.window.exercise_combo.findData(exercise_type)
        if index >= 0:
            self.window.exercise_combo.setCurrentIndex(index)

        # --- BUFOR NA PODSUMOWANIA ---
        self.repetition_buffer = []

        # --- WĄTKI I KOLEJKI ---
        self.queue_a = queue.Queue(maxsize=2)
        self.queue_b = queue.Queue(maxsize=2)

        self.inference_worker = SyncInferenceWorker(self.queue_a, self.queue_b)

        # Przekazanie typu ćwiczenia do workera (wymaga uwzględnienia "self.exercise_type" wewnątrz SyncInferenceWorker)
        self.inference_worker.exercise_type = exercise_type

        self.inference_worker.frames_ready.connect(self.window.update_both_labels)
        self.inference_worker.evaluation_ready.connect(self.handle_evaluation_result)
        self.inference_worker.start()

        self.worker_a = CameraWorker(0, self.queue_a)
        self.worker_a.start()

        self.worker_b = None
        self.old_workers = []

        self.window.connect_btn.clicked.connect(self.connect_ip_camera)
        if hasattr(self.window, "save_settings_btn"):
            self.window.save_settings_btn.clicked.connect(self.save_current_settings)

    def save_current_settings(self):
        self.settings["ip_camera"] = self.window.ip_input.text()

        # Pobranie typu ćwiczenia z ComboBox
        selected_exercise = self.window.exercise_combo.currentData()
        self.settings["exercise_type"] = selected_exercise

        # Natychmiastowa aktualizacja w dziale inferencji
        if hasattr(self, 'inference_worker'):
            self.inference_worker.exercise_type = selected_exercise

        try:
            self.settings["log_retention_days"] = int(self.window.retention_input.text())
            self.settings["summary_frequency"] = int(self.window.summary_input.text())
            self.summary_frequency = self.settings["summary_frequency"]
        except ValueError:
            pass
        save_settings(self.settings)

    def connect_ip_camera(self):
        self.save_current_settings()

        if self.worker_b is not None:
            self.worker_b._is_running = False
            self.old_workers.append(self.worker_b)

        stream_url = self.window.ip_input.text()
        self.worker_b = CameraWorker(stream_url, self.queue_b)
        self.worker_b.start()

    def handle_evaluation_result(self, result):
        if not result:
            return

        # 1. Błyskawiczny podgląd na bieżące (pojedyncze) powtórzenie w UI
        formatted_single_rep = self.window.format_repetition_details(result)
        self.window.result_details_console.setPlainText(formatted_single_rep)

        # 2. Dodajemy do bufora
        self.repetition_buffer.append(result)

        # 3. Jeśli zebraliśmy wystarczająco dużo powtórzeń, robimy podsumowanie
        if len(self.repetition_buffer) >= self.summary_frequency:
            self.generate_and_save_summary()

    def generate_and_save_summary(self):
        if not self.repetition_buffer:
            return

        # --- OBLICZENIA DANYCH DO PODSUMOWANIA ---
        scores = [r.get("score", 0.0) for r in self.repetition_buffer]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0

        all_errors = []
        for r in self.repetition_buffer:
            for e in r.get("errors", []):
                # Pobieramy treść błędu
                all_errors.append(e.get("message", "Nieznany błąd"))

        if all_errors:
            # Wyciąga najczęściej występujący string z listy błędów
            most_common_error = Counter(all_errors).most_common(1)[0][0]
        else:
            most_common_error = "Brak większych błędów - świetna robota!"

        exercise_type = self.repetition_buffer[-1].get("exercise_type", "Nieznane")
        count = len(self.repetition_buffer)

        # --- BUDOWANIE TEKSTU ---
        summary_text = (
            f"➤ Seria: {count} powt. | Ćwiczenie: {exercise_type}\n"
            f"➤ Średni wynik techniki: {avg_score} pkt\n"
            f"➤ Główny element do poprawy:\n   {most_common_error}\n"
            f"--------------------------------------------------"
        )

        # Aktualizacja okna asystenta
        self.window.log_console.append(summary_text)

        # Zapis logów
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"=== [{now_str}] ZESTAWIENIE SERII ===\n{summary_text}\n\n"

        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Nie udało się zapisać do pliku: {e}")

        # Resetujemy bufor przed kolejną serią
        self.repetition_buffer.clear()

    def cleanup(self):
        self.save_current_settings()

        # Wyrzucamy niedokończoną serię do pliku, żeby nie ucięło danych na sam koniec
        if self.repetition_buffer:
            self.generate_and_save_summary()

        self.worker_a.stop()
        if self.worker_b is not None:
            self.worker_b.stop()
        self.inference_worker.stop()


def main():
    app = QApplication(sys.argv)
    controller = AppController()
    controller.window.show()

    app.aboutToQuit.connect(controller.cleanup)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()