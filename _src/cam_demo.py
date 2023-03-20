import time
import dlib
import cv2
from eye_detector import EyesDetector
from pupil_detector import PupilsDetector
from sota import PupilsDetectorSOTA
import utils


class Cam:
	def __init__(self):
		self.eye_detector = EyesDetector()
		self.pupil_detector = PupilsDetector()
		self.pupil_detector.load('../artifacts/pupils_detector.dat')
		self.sota = PupilsDetectorSOTA()

	def run(self, model='ert', show_eye_rect=False, show_fps=False):
		cap = cv2.VideoCapture(0)
		if show_fps:
			frame_count = 0
			start_time = time.time()
		while True:
			ret, image = cap.read()
			if model == 'ert':
				self.__detect_ert(image, show_eye_rect)
			else:
				self.__detect_sota(image)
			if show_fps:
				frame_count += 1
				elapsed_time = time.time() - start_time
				fps = str(self.__calc_fps(frame_count, elapsed_time))
				utils.draw_text(image, fps)
			cv2.imshow('Cam', image)
			key = cv2.waitKey(1)
			if key == ord('q'):
				break
		cv2.destroyAllWindows()

	def __detect_sota(self, image):
		pupils = self.sota.predict(image)
		if pupils:
			for pup in pupils:
				utils.draw_circle(image, (pup[0], pup[1]))

	def __detect_ert(self, image, show_eye_rect=False):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			eyes = self.eye_detector.predict(image)
			if eyes:
				for eye in eyes:
					(x1, y1, x2, y2) = eye[0], eye[1], eye[2], eye[3]
					if show_eye_rect:
						utils.draw_rect(image, (x1, y1), (x2, y2), color=(0, 255, 0))
					rect = dlib.rectangle(x1, y1, x2, y2)
					x, y = self.pupil_detector.predict(gray, rect)
					utils.draw_circle(image, (x, y))

	def __calc_fps(self, frame_count, elapsed_time):
		return int(frame_count / elapsed_time)


if __name__ == '__main__':
	cam = Cam()
	cam.run(model='ert', show_eye_rect=False, show_fps=True)
