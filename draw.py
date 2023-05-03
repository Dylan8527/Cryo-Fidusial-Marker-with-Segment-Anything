import cv2
import numpy as np

LEFT_CLICK = 1
RIGHT_CLICK = 0
FOREGROUND_CIRCLE_COLOR = (0, 255, 0) # Green, 1, Left click
BACKGROUND_CIRCLE_COLOR = (0, 0, 255) # Red,   0, Right click

def draw_circles(display_image, point_coords, point_labels):
    '''
    :param display_image:   (H, W, 3) array of cv2 Image
    :param points:          (N, 2) array of points
    :param labels:          (N, ) array of labels
    :return:                display_image
    '''

    for point_coord, point_label in zip(point_coords, point_labels):
        if point_label == LEFT_CLICK:
            circle_color = FOREGROUND_CIRCLE_COLOR
        elif point_label == RIGHT_CLICK:
            circle_color = BACKGROUND_CIRCLE_COLOR
        else:
            raise ValueError(f"Unknown point label {point_label}")

        cv2.circle(display_image, tuple(point_coord), 5, circle_color, -1)

    return display_image

def draw_single_mask(display_image, mask, color, alpha=0.5):
    '''
    :param display_image:   (H, W, 3) array of cv2 Image
    :param mask:            (H, W) array of mask
    :param alpha:           float in [0, 1], transparency of mask
    :return:                display_image
    '''
    mask = mask.astype(np.float32)
    mask = np.stack([mask, mask, mask], axis=2)
    display_image = display_image.astype(np.float32)
    display_image = np.where(mask == 1, display_image * (1 - alpha) + alpha * color, display_image)
    display_image = display_image.astype(np.uint8)

    return display_image
