'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2023-05-03 21:30:08
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2023-05-04 00:18:51
FilePath: \Cryo-Fidusial-Marker-with-Segment-Anything\sam.py
Description: Use sam to generate mask from points

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import cv2
import time
import warnings
import numpy as np
from draw import *
from segment_anything import sam_model_registry, SamPredictor

ADD_POINT_STATE = 0
GENERATE_MASK_STATE = 1
MODEL_PATH = 'Pretrained_models\\sam_vit_h_4b8939.pth'
# IMAGE_PATH = 'Figures\\average_micrograph.png'
IMAGE_PATH = 'Figures\\dog.jpg'
WINDOW_NAME = "Sam Mask Generator"

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

def mouse_callback(event, x, y, flags, param):
    global point_coords, point_labels, display_state
    if display_state == ADD_POINT_STATE:
        if event == cv2.EVENT_LBUTTONDOWN:
            point_coords.append([x, y])
            point_labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            point_coords.append([x, y])
            point_labels.append(0)

    elif display_state == GENERATE_MASK_STATE:
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            warnings.warn("Cannot add points when generating mask.")

def reset():
    global point_coords, point_labels, mask_input, all_masks, all_mask_inputs, all_scores, display_state, manually_chosen_best_mask_idx
    point_coords = []
    point_labels = []
    mask_input = None
    display_state = ADD_POINT_STATE
    all_masks = None
    all_mask_inputs = None  
    all_scores = None
    manually_chosen_best_mask_idx = -1

# 1. Load the model1
sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATH).cuda() # cuda:0 3090 actually
predictor = SamPredictor(sam)

# 2. Load the image
unmasked_image = cv2.imread(IMAGE_PATH)
print(unmasked_image.shape)
print(f'Min pixel value: {unmasked_image.min()} | Max pixel value: {unmasked_image.max()}')
if unmasked_image.shape[0] > 512 or unmasked_image.shape[1] > 512: # Avoid too large image
    unmasked_image = cv2.resize(unmasked_image, (512, 512))
# unmasked_image = cv2.cvtColor(unmasked_image, cv2.COLOR_BGR2RGB)

point_coords = []
point_labels = []
mask_input = None
display_state = 0 # 0: add points, 1: generate mask
all_masks = None
all_mask_inputs = None
all_scores = None
manually_chosen_best_mask_idx = -1
random_rgb_color1, random_rgb_color2 = np.random.randint(0, 255, size=(3, )), np.random.randint(0, 255, size=(3, ))

def smooth_transition(color1, color2):
    ratio = (round(time.time() * 1000 )) % 1000 / 1000
    interpolated_color = (1 - ratio) * color1 + ratio * color2
    return interpolated_color.astype(np.uint8)

# 3. Create cv2 window
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

helper_info0 = 'Add Point | Foreground: Left Click | Background: Right Click | Generate Mask: Space | Reset: R | Exit: Esc'
helper_info1 = 'Generate Mask | Add Point: Space | Left Mask: A | Right Mask: D | Exit: Esc'

while True:
    display_image = unmasked_image.copy()
    # helper_info = helper_info0 if display_state == ADD_POINT_STATE else helper_info1
    if manually_chosen_best_mask_idx == -1 or display_state == ADD_POINT_STATE:
        helper_info = 'Fore: Left Clk | Back: Right Clk | Gen Mask: Space | Reset: R'
    else:
        helper_info = f'Total Masks: {len(all_masks)} | Current Mask Idx: {manually_chosen_best_mask_idx} | Score {all_scores[manually_chosen_best_mask_idx]:.4f}'
    
    cv2.putText(display_image, helper_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # Draw points
    display_image = draw_circles(display_image, point_coords, point_labels)
    # Draw single mask
    if manually_chosen_best_mask_idx != -1:
        smooth_varied_rgb_color = smooth_transition(random_rgb_color1, random_rgb_color2)
        display_image = draw_single_mask(display_image, all_masks[manually_chosen_best_mask_idx], smooth_varied_rgb_color, alpha=0.5)

    cv2.imshow(WINDOW_NAME, display_image)
    key = cv2.waitKey(1)

    if key == ord('r'):
        reset()
    elif key == ord(' '):
        display_state = 1 - display_state

        if len(point_coords) > 0:
            np_point_coords, np_point_labels = np.array(point_coords), np.array(point_labels)
            masks, scores, logits = generate_mask(
                point_coords=np_point_coords,
                point_labels=np_point_labels,
                mask_input=mask_input,
                unmasked_image=unmasked_image,
                predictor=predictor
            )
            
            point_coords = []
            point_labels = []
            
            # Since Sam generate multiple masks, we need to choose the best one manually
            manually_chosen_best_mask_idx = np.argmax(scores)

            all_masks = masks
            all_mask_inputs = logits
            all_scores = scores

    elif key == ord('a'):
        if manually_chosen_best_mask_idx != -1:
            manually_chosen_best_mask_idx = max(0, manually_chosen_best_mask_idx - 1)

    elif key == ord('d'):
        if manually_chosen_best_mask_idx != -1:
            manually_chosen_best_mask_idx = min(len(all_masks) - 1, manually_chosen_best_mask_idx + 1)

    elif key == 27:
        break
    elif key == ord('q') and len(point_coords) > 0:
        point_coords.pop()
        point_labels.pop()
        

    






    



    











