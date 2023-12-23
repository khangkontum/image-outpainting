import random
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

def gen_local_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    is_slide_x = random.random() > 0.5
    if is_slide_x:
        offset_x = random.randint(0, mask_w - harea_w)
        offset_y = 0
    else:
        offset_x = 0
        offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))


def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin: ymin + h, xmin: xmin + w]