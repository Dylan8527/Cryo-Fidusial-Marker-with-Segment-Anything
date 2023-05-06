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

import numpy as np
from functools import reduce

def generate_mask(point_coords, point_labels, mask_input, unmasked_image, predictor):
    '''
    :param points:      (N, 2) array of points
    :param labels:      (N, ) array of labels
    :param logit_mask:  (H, W) array of logits
    :param image:       (H, W, 3) array of image
    :param predictor:   SamPredictor
    :return:            masks, scores, logits
    '''

    predictor.set_image(unmasked_image)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=mask_input[None, ...] if mask_input is not None else None,
        multimask_output=True
    )
    return masks, scores, logits

def merge_masks_scores_mask_inputs(all_masks, all_scores, all_mask_inputs):
    '''
    :param all_masks:       (N, H, W) array of masks
    :param all_scores:      (N, ) array of scores
    :param all_mask_inputs: (N, H, W) array of mask inputs
    :param points:          (M, 2) array of points
    :param labels:          (M, ) array of labels
    :return:                 merged_mask, merged_score, merged_mask_input
    '''
    all_masks = [reduce(lambda x, y: np.logical_or(x, y), all_masks)]
    all_scores = [1.14514]
    all_mask_inputs = [reduce(lambda x, y: np.logical_or(x, y), all_mask_inputs)]

    return all_masks, all_scores, all_mask_inputs