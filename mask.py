from pycocotools import mask as mask_utils

from functools import reduce
import numpy as np
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
    all_mask_inputs = [reduce(lambda x, y: np.logical_or(x, y), all_mask_inputs)] if all_mask_inputs is not None else None

    return all_masks, all_scores, all_mask_inputs

def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle