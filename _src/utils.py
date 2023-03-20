import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# model usage utils
LEFT_EYE_POINTS = [0, 1, 2, 3, 4, 5]
RIGHT_EYE_POINTS = [6, 7, 8, 9, 10, 11]
LEFT_EYE_LEFT_EDGE = 0
LEFT_EYE_RIGHT_EDGE = 3
RIGHT_EYE_LEFT_EDGE = 6
RIGHT_EYE_RIGHT_EDGE = 9


def shape_to_np(shape, dtype=int):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def is_pupil_in_eye_box(pupil_coord, eye):
    px, py = pupil_coord
    contains = False
    ex1, ey1, ex2, ey2 = eye
    if ex2 > px > ex1 and ey2 > py > ey1:
        contains = True
    return contains


# model evaluation utils
def err_plot(data, save=False, plot_path='./plot.png'):
    plt.plot(data)
    if save:
        plt.savefig(plot_path)
    plt.show()
    plt.savefig(sys.stdout.buffer)
    sys.stdout.flush()


# draw on image utils
def draw_points(image, landmarks):
    for l in landmarks:
        draw_circle(image, (l[0], l[1]), radius=0)


def draw_rect(image, start_coord, end_coord, color=(0, 0, 255), thickness=2):
    cv2.rectangle(image, (start_coord[0], start_coord[1]), (end_coord[0], end_coord[1]), color, thickness)


def draw_circle(image, center, radius=3, color=(0, 0, 255), thickness=1):
    cv2.circle(image, center, radius, color, thickness)


def draw_plus(image, center, length=10, color=(0, 0, 255), thickness=2):
    cv2.line(image, (center[0] - length, center[1]), (center[0] + length, center[1]), color, thickness)
    cv2.line(image, (center[0], center[1] - length), (center[0], center[1] + length), color, thickness)


def draw_text(image, text, ord=(7, 70), font=cv2.FONT_HERSHEY_SIMPLEX, font_size=3, color=(100, 255, 0), thickness=3, line_type=cv2.LINE_AA):
    cv2.putText(image, text, ord, font, font_size, color, thickness, line_type)
