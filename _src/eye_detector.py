import dlib
import utils


class EyesDetector:
    def __init__(self):
        self.__face_detector = dlib.get_frontal_face_detector()
        self.__landmarks_detector = dlib.shape_predictor('../artifacts/eye_detector.dat')
        self.__left_eye_points = utils.LEFT_EYE_POINTS
        self.__right_eye_points = utils.RIGHT_EYE_POINTS

    def predict(self, image):
        rects = self.__face_detector(image, 1)
        if rects:
            for rect in rects:
                landmarks = self.__landmarks_detector(image, rect)
                landmarks = utils.shape_to_np(landmarks)
                lsx, lsy, lex, ley = self.__get_single_eye_rect(landmarks[utils.LEFT_EYE_LEFT_EDGE], landmarks[
                    utils.LEFT_EYE_RIGHT_EDGE])
                rsx, rsy, rex, rey = self.__get_single_eye_rect(landmarks[utils.RIGHT_EYE_LEFT_EDGE], landmarks[
                    utils.RIGHT_EYE_RIGHT_EDGE])
            return [(lsx, lsy, lex, ley), (rsx, rsy, rex, rey)]

    def __get_single_eye_rect(self, left_edge_coord, right_edge_coord, add=10):
        dx = right_edge_coord[0] - left_edge_coord[0]
        dy = right_edge_coord[1] - left_edge_coord[1]
        width = int(dx + add * 2)
        cy = dy / 2
        sx = int(left_edge_coord[0] - add)
        sy = int((left_edge_coord[1] + cy) - (0.5 * width))
        ex = sx + width
        ey = sy + width
        return sx, sy, ex, ey
