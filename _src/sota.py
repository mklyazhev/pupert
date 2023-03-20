import numpy as np
import cv2
import dlib
import utils


# use this class for detect pupils to make labels for train data
class PupilsDetectorSOTA:
    def __init__(self, threshold=60):
        self.__face_detector = dlib.get_frontal_face_detector()
        self.__landmarks_detector = dlib.shape_predictor('../artifacts/eye_detector.dat')
        self.__left_eye_points = utils.LEFT_EYE_POINTS
        self.__right_eye_points = utils.RIGHT_EYE_POINTS
        self.__kernel = np.ones((9, 9), np.uint8)
        self.__threshold = threshold

    def predict(self, image):
        rects = self.__face_detector(image, 1)
        if rects:
            for rect in rects:
                landmarks = self.__landmarks_detector(image, rect)
                landmarks = utils.shape_to_np(landmarks)
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask = self.__eye_on_mask(landmarks, mask, self.__left_eye_points)
                mask = self.__eye_on_mask(landmarks, mask, self.__right_eye_points)
                mask = cv2.dilate(mask, self.__kernel, 5)
                eyes = cv2.bitwise_and(image, image, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = (landmarks[3][0] + landmarks[6][0]) // 2
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                threshold = self.__threshold
                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                thresh = cv2.erode(thresh, None, iterations=2)
                thresh = cv2.dilate(thresh, None, iterations=4)
                thresh = cv2.medianBlur(thresh, 3)
                thresh = cv2.bitwise_not(thresh)
                left_coord = self.__contouring(thresh[:, 0:mid], mid)
                right_coord = self.__contouring(thresh[:, mid:], mid, True)
                if left_coord and right_coord:
                    # [(lx, ly), (rx, ry)]
                    return [left_coord, right_coord]

    def __eye_on_mask(self, shape, mask, side):
        points = [shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)
        return mask

    def __contouring(self, thresh, mid, right=False):
        try:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if right:
                cx += mid
            return (cx, cy)
        except:
            return
