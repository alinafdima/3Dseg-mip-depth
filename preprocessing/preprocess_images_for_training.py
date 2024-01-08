# ------------------------------------------------------
# File: preprocess_images_for_training.py
# Author: Alina Dima <alina.dima@tum.de>
#
# Preprocessing steps that the images undergo before being fed to the network
# Step 1: Clip images to the range [-150, 250] HU and save them as 8-bit images
# Step 2: Pre-crop each img from a size of [512, 512, ?] to [300, 300, 150] around the ostium of the superior mesenteric artery
# The pre-cropped images are afterwards fed to the data loader, which augments the images and further crops them to a size of [256, 256, 128].
# ------------------------------------------------------

from typing import Dict, Tuple

import numpy as np
import nibabel as nib


# #############################################
# Step 1
# #############################################

def clip_image_custom_rescaling(img: np.ndarray, clip_values : Tuple[int, int]) -> np.ndarray:
    """
    Clips an image to the range [clip_min, clip_max] and rescales it to the range [0, 255].

    Args:
        img (np.ndarray): input image
        clip_values (Tuple[int, int]): min and max clip values

    Returns:
        np.ndarray: Clipped array rescaled to the range [0, 255]
    """    
    clip_min, clip_max = clip_values
    img = np.clip(img, clip_min, clip_max)
    img = (img - clip_min) / (clip_max - clip_min) * 255
    return img.astype('uint8')


def clip_image_ct_default(img):    
    return clip_image_custom_rescaling(img, (-150, 250))


def convert_single_img_to_8bit(input_path, output_path):
    img_nib = nib.load(input_path)
    img_clipped = clip_image_ct_default(img_nib.get_fdata())
    nib.save(nib.Nifti1Image(img_clipped, img_nib.affine), output_path)


# #############################################
# Step 2
# #############################################

def crop_image(x : np.ndarray, crop_bounds: Dict) -> np.ndarray:
    """
    Crops a numpy array according to the crop boundaries
    Example crop_bounds: { "x": [100, 400], "y": [150, 450], "z": [100, 250] }

    Args:
        x (np.ndarray): numpy array to be cropped.
        crop_bounds (Dict): a dictionary containing the crop indices for each axis.

    Returns:
        np.ndarray: cropped numpy array.
    """    
    slice_object = tuple([slice(*x) for x in crop_bounds.values()])
    return x[slice_object]


def crop_image_around_sma_ostium(img: np.ndarray, crop_center: np.ndarray) -> np.ndarray:
    """
    Starting from the crop center, computes the crop indices for an input image.
    If the crop indices fall outside of the image bounds, the crop is shifted to not alter the size of the output image.
    Each image is cropped to a size of [300, 300, 150].
    The crop center is the ostium of the superior mesenteric area in the pixel coordinate space (origin at (0, 0, 0) - image left corner).

    Args:
        img (np.ndarray): numpy array to be cropped.
        crop_center (np.ndarray): a 3D vector containing the crop center coordinates in pixel space.

    Returns:
        Dict: A dictionary containing the crop indices for an input image.
    """    
    crop_sizes = dict(zip('xyz', [300, 300, 150]))
    img_size = img.shape

    crop_bounds = {}
    for axis, axis_name in zip([0, 1, 2], 'xyz'):
        crop_size = crop_sizes[axis_name]
        axis_size = img_size[axis]
        seedpoint_axis = crop_center[axis]

        axis_min, axis_max = (seedpoint_axis - crop_size // 2, seedpoint_axis + crop_size // 2)
        if axis_min < 0:
            axis_min, axis_max = 0, crop_size

        elif axis_max > axis_size: 
            axis_min, axis_max = axis_size - crop_size, axis_size

        crop_bounds[axis_name] = (axis_min, axis_max)

    return crop_image(img, crop_bounds)