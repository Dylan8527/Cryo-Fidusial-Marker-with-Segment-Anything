'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2023-05-03 21:30:08
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2023-05-06 18:23:20
FilePath: \Cryo-Fidusial-Marker-with-Segment-Anything\sam.py
Description: Use sam to generate mask from points

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import cv2
import time
import warnings

import numpy as np
from draw import *
from file import *
from mask import *
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

ADD_POINT_STATE = 0
GENERATE_MASK_STATE = 1
MODEL_PATH = 'Pretrained_models\\sam_vit_h_4b8939.pth'
# IMAGE_PATH = 'Figures\\average_micrograph.png'
# IMAGE_PATH = 'Figures\\dog.jpg'
IMAGE_PATH = None
SAVE_PATH = None
WINDOW_NAME = "Sam Mask Generator"

if IMAGE_PATH is None:
    IMAGE_PATH = select_image_path()
    
class MaskGenerator:
    def __init__(self):
        self.point_coords = []
        self.point_labels = []
        self.mask_input = None
        self.display_state = ADD_POINT_STATE
        self.all_masks = None
        self.all_mask_inputs = None  
        self.all_scores = None
        self.manually_chosen_best_mask_idx = -1
    
        self.random_rgb_color1, self.random_rgb_color2 = np.random.randint(0, 255, size=(3, )), np.random.randint(0, 255, size=(3, ))

    def reset(self):
        self.point_coords = []
        self.point_labels = []
        self.mask_input = None
        self.display_state = ADD_POINT_STATE
        self.all_masks = None
        self.all_mask_inputs = None  
        self.all_scores = None
        self.manually_chosen_best_mask_idx = -1
        
    def smooth_transition(self):
        ratio = (round(time.time() * 1000 )) % 1000 / 1000
        interpolated_color = (1 - ratio) * self.random_rgb_color1 + ratio * self.random_rgb_color2
        return interpolated_color.astype(np.uint8)

    def flip_state(self):
        self.display_state = 1 - self.display_state



    def generate_mask(self, unmasked_image, predictor, mask_generator):
        '''
        :param unmasked_image:       (H, W, 3) array of image
        :param predictor:            SamPredictor
        :param mask_generator:       SamAutomaticMaskGenerator
        '''
        if len(self.point_coords) == 0:
            self.generate_all_masks(unmasked_image, mask_generator)
        else:
            predictor.set_image(unmasked_image)
            np_point_coords, np_point_labels = np.array(self.point_coords), np.array(self.point_labels)
            self.all_masks, self.all_scores, self.all_mask_inputs = predictor.predict(
                point_coords=np_point_coords,
                point_labels=np_point_labels,
                mask_input=self.mask_input[None, ...] if self.mask_input is not None else None,
                multimask_output=True
            )

        # Since Sam generate multiple masks, we need to choose the best one manually
        MaskGen.autoset_best_mask()
        MaskGen.reset_point() 

    def generate_all_masks(self, unmasked_image, mask_generator):
        '''
        :param unmasked_image:       (H, W, 3) array of image
        :param mask_generator:       SamAutomaticMaskGenerator
        '''
        warnings.warn("Since no points are added, generate masks for an entire image.")
        masks_list = mask_generator.generate(unmasked_image) # each element is a dict, we get the segmentataion in dict
        self.all_masks  = [m["segmentation"] for m in masks_list]
        self.all_scores = [m["predicted_iou"] for m in masks_list]
        self.all_mask_inputs = None

    def autoset_best_mask(self):
        if self.all_scores is not None:
            self.manually_chosen_best_mask_idx = np.argmax(self.all_scores)

    def reset_point(self):
        self.point_coords = []
        self.point_labels = []

    def have_mask(self):
        return self.all_masks is not None

    def move_to_left_mask(self):
        if self.have_mask():
            self.manually_chosen_best_mask_idx = (self.manually_chosen_best_mask_idx - 1) % len(self.all_masks)

    def move_to_right_mask(self):
        if self.have_mask():
            self.manually_chosen_best_mask_idx = (self.manually_chosen_best_mask_idx + 1) % len(self.all_masks)

    def pop_point(self):
        if len(self.point_coords) > 0:
            self.point_coords.pop()
            self.point_labels.pop()

    def automerge_all_masks(self):
        if self.all_masks is not None and len(self.all_masks) > 1:
            self.all_masks, self.all_scores, self.all_mask_inputs = merge_masks_scores_mask_inputs(
                all_masks=self.all_masks[1:],
                all_scores=self.all_scores[1:],
                all_mask_inputs=self.all_mask_inputs[1:] if self.all_mask_inputs is not None else None
            )
            self.manually_chosen_best_mask_idx = 0
            print(self.all_masks[0].shape)
            self.mask_input = self.all_masks[0]

    def mouse_callback(self, event, x, y, flags, param):
        # global point_coords, point_labels, display_state
        # if display_state == ADD_POINT_STATE:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point_coords.append([x, y])
            self.point_labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.point_coords.append([x, y])
            self.point_labels.append(0)

# 1. Load the model1
sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATH).cuda() # cuda:0 3090 actually
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

# 2. Load the image
unmasked_image = cv2.imread(IMAGE_PATH)
print(unmasked_image.shape)
print(f'Min pixel value: {unmasked_image.min()} | Max pixel value: {unmasked_image.max()}')
if unmasked_image.shape[0] > 512 or unmasked_image.shape[1] > 512: # Avoid too large image
    unmasked_image = cv2.resize(unmasked_image, (512, 512))
# unmasked_image = cv2.cvtColor(unmasked_image, cv2.COLOR_BGR2RGB)

MaskGen = MaskGenerator()

# 3. Create cv2 window
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, MaskGen.mouse_callback)

helper_info0 = 'Add Point | Foreground: Left Click | Background: Right Click | Generate Mask: Space | Reset: R | Exit: Esc'
helper_info1 = 'Generate Mask | Add Point: Space | Left Mask: A | Right Mask: D | Exit: Esc'

while True:
    display_image = unmasked_image.copy()
    # helper_info = helper_info0 if display_state == ADD_POINT_STATE else helper_info1
    if not MaskGen.have_mask() or MaskGen.display_state == ADD_POINT_STATE:
        helper_info = 'Fore: Left Clk | Back: Right Clk | Gen Mask: Space | Reset: R'
    else:
        helper_info = f'Total Masks: {len(MaskGen.all_masks)} | Current Mask Idx: {MaskGen.manually_chosen_best_mask_idx} | Score {MaskGen.all_scores[MaskGen.manually_chosen_best_mask_idx]:.4f}'
    
    cv2.putText(display_image, helper_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    display_image = draw_circles(display_image, MaskGen.point_coords, MaskGen.point_labels) # Draw points

    if MaskGen.have_mask(): 
        display_image = draw_single_mask(display_image, MaskGen.all_masks[MaskGen.manually_chosen_best_mask_idx], MaskGen.smooth_transition(), alpha=0.5) # Draw single mask

    cv2.imshow(WINDOW_NAME, display_image)
    key = cv2.waitKey(1)

    if key == ord('r'):
        MaskGen.reset()
    elif key == ord(' '):
        MaskGen.flip_state()
        if MaskGen.display_state == GENERATE_MASK_STATE:
            MaskGen.generate_mask(unmasked_image, predictor, mask_generator)    

    elif key == ord('a'):
        MaskGen.move_to_left_mask()

    elif key == ord('d'):
        MaskGen.move_to_right_mask()

    elif key == 27:
        break

    elif key == ord('q'):
        MaskGen.pop_point()

    elif key == ord('m'):
        MaskGen.automerge_all_masks()
            
    elif key == ord('s'):
        pass

        

    






    



    











