import queue
import sys

from PyQt5.QtWidgets import QApplication

from backend import CameraWorker, SyncInferenceWorker
from ui import MainWindow

VIDEO_SOURCE = "/home/arczi/Projects/ProjektJugger/KCK_Sentymentalne_Niziolki/references/segmenty/3/bez_kroki_3b.mp4"


# VIDEO_SOURCE = 0


class AppController:
	def __init__(self):
		self.window = MainWindow()

		self.queue_a = queue.Queue(maxsize=2)
		self.queue_b = queue.Queue(maxsize=2)

		self.camera_a = CameraWorker(VIDEO_SOURCE, self.queue_a)

		self.inference_worker = SyncInferenceWorker(
			self.queue_a,
			self.queue_b,
			exercise_type="arms_only",
			target_repetitions=5,
			dominant_side="right",
			model_path="yolov8n-pose.pt",
		)

		self.inference_worker.frames_ready.connect(self.window.update_both_labels)
		self.inference_worker.evaluation_ready.connect(self.window.append_evaluation_result)
		self.inference_worker.debug_ready.connect(self.window.update_debug_status)

	def start(self):
		self.camera_a.start()
		self.inference_worker.start()

	def stop(self):
		self.camera_a.stop()
		self.inference_worker.stop()


def main():
	app = QApplication(sys.argv)

	controller = AppController()
	controller.window.show()
	controller.start()

	app.aboutToQuit.connect(controller.stop)

	sys.exit(app.exec_())


if __name__ == "__main__":
	main()
