# ------------------------------------------------------
# File: preprocess_images_for_projection.py
# Author: Alina Dima <alina.dima@tum.de>
#
# The entire proces of generating the MIP projections from the original arterial-phase CT images
# ------------------------------------------------------

from typing import Tuple, Dict

import cc3d
import numpy as np
import torch
from skimage import morphology

import affine_utils
from preprocessing.preprocess_images_for_training import crop_image_around_sma_ostium


def center_crop_tensor(im, crop_size):
    """
    Crops a tensor to a given size around its center.
    If the crop_size is odd, the extra pixel is going to be on the right side.

    Args:
        im (torch/numpy tensor): k-dimensional tensor
        crop_size (numpy array): output shape, as a k-length array

    Raises:
        ValueError: if the crop size does not match the dimensionaliy of the input size

    Returns:
        tensor: cropped tensor
    """    
    im_center = np.array(im.shape) // 2

    if len(im_center) != len(crop_size):
        raise ValueError('Crop size and image size must have the same number of dimensions')

    left_offset = crop_size // 2
    right_offset = crop_size - left_offset

    slice_object = tuple(slice(l, r) for l, r in zip(im_center - left_offset, im_center + right_offset))
    return im[slice_object]



def threshold_renormalize(img, threshold):
    img_th = np.clip(img, threshold, 255)
    img_th = (img_th - threshold) / (255 - threshold)
    return img_th


def dilate_vertebrae_mask(vertebrae_mask_anduin):
    """
    Given a vertebrae mask computed using anduin  https://anduin.bonescreen.de), 
    dilates the mask to ensure all hyperintense pixels of the vertebra are removed from the image.

    mask_filename = f'{prefix}_seg-vert_msk.nii.gz'

    Args:
        vertebrae_mask_anduin (np.ndarray): Input vertebrae mask (0: background, 1: vertebrae)

    Returns:
        np.ndarray: dilated vertebrae mask
    """    
    mask_dilated = morphology.binary_dilation(vertebrae_mask_anduin, footprint=morphology.cube(3))
    new_vert_mask = (mask_dilated == 0).astype('int')
    return new_vert_mask


def compute_rib_mask(img_12bit, vert_mask, rib_threshold=300, cc_cutoff=50000):
    """
    Computes the rib mask from a CT image and a vertebrae mask using a series of morphological operations.
    
    Args:
        img_12bit (np.ndarray): raw CT image (12-bit, no intensity preprocessing), cropped around the SMA ostium.
        vert_mask (np.ndarray): vertebrae mask (0: background, 1: vertebrae), cropped around the SMA ostium.
        rib_threshold (int, optional): The threshold for clipping the CT image. Defaults to 300. Fixed for the entire dataset.
        cc_cutoff (int, optional): The cutoff below which the connected components are removed. Defaults to 50000. Varies from case to case, sometimes higher (100000).

    Returns:
        np.ndarray: image with the ribs, vertebrae and other tissue removed.
    """

    # Clip the image
    clipped_img = (img_12bit > rib_threshold).astype('int') * vert_mask

    # Compute connected components
    cc = cc3d.connected_components(clipped_img, connectivity=6)
    label_list = [(x, np.sum(cc[cc == x])) for x in np.unique(cc)]

    # Remove small connected components, then perform erosion to identify the aorta segments
    sorted_label_list = sorted(label_list, key=lambda x:x[1])
    large_labels = [x[0] for x in sorted_label_list if x[1] > cc_cutoff]
    largest_components = np.zeros_like(cc)
    for label in large_labels:
        largest_components[cc == label] = 1
    aorta_trace = morphology.binary_erosion(largest_components, morphology.ball(7))
    aorta_labels = np.unique(cc[aorta_trace == 1])

    # Construct rib mask by taking all of the large connected components that are not part of the aorta
    rib_mask = np.zeros_like(cc)
    for label in large_labels:
        if label not in aorta_labels:
            rib_mask[cc == label] = 1
    rib_mask_dilated = morphology.binary_dilation(rib_mask , footprint=morphology.cube(5)).astype('int')
    
    return (1 - rib_mask_dilated)


def preprocess_img(img_12bit, vertebrae_mask_anduin):
    """
    Preprocesses a raw 12-bit arterial-phase CT image to remove the ribs, vertebrae and other small tissue.

    Args:
        img_12bit (np.ndarray): cropeed raw CT image (12-bit, no intensity preprocessing), cropped around the SMA ostium.
        vertebrae_mask_anduin (np.ndarray): cropped vertebrae segmentation from anduin

    Returns:
        np.ndarray: preprocessed image, ready to be projected into a MIP.
    """    
    # Already cropped around the SMA ostium
    img = threshold_renormalize(img_12bit, 150)
    vert_mask = dilate_vertebrae_mask(vertebrae_mask_anduin)
    rib_mask = compute_rib_mask(img_12bit, vert_mask)

    # Apply the masks
    img = img * vert_mask * rib_mask
    
    # Remove the lung vessels
    img[:, :40, :] = 0
    img[:22, :22, :11] = 0
    img[-22:, -22:, -11:] = 0

    return img


def project_img_bidirectional(img):
    """
    Computes forward and backward depth-encoded MIP projections from a 3D CT image, while also returning the depth information.
    The projections are used for manual annotation of the 3D CT images, while the depth maps are used to project the annotations back into the 3D space.
    
    The projections are always done along the y-axis (axis=1). The reason for this is generalizability of the 
    projection process: the image is first rotated to whatver orientation is needed, and then the projection is done as a simple max operation.

    The input image is pre-processed to remove the ribs and vertebrae, and has a shape of (256, 256, 128).
    
    The 3D depth volumes: depth_fwd and depth_bwd are the depth maps for the forward and backward projections, respectively. 
    They are used to project 2D annotations back into the 3D space.
    Their range is [0, 255] and have the shape of the input image (256, 256, 128).
    
    All 3 2D projections (img_mip, projection_fwd, projection_bwd) are normalized to the range [0, 1] and
    have the shape (256, 128).

    Args:
        img (np.ndarray): Input image (3D CT image), of size (256, 256, 128).

    Returns:
        Dict: The MIP projections and depth volumes. 
    """    
    img_mip = np.max(img, axis=1)
    depth_fwd = np.argmax(img, axis=1)
    depth_bwd_inverted = np.argmax(img[:, ::-1, :], axis=1)
    depth_bwd = 255 - depth_bwd_inverted

    mip_dict = {
        'img_mip': img_mip,
        'projection_fwd': np.sqrt(img_mip / 255.) * (depth_fwd / 255.),
        'projection_bwd': np.sqrt(img_mip / 255.) * (depth_bwd_inverted / 255.),
        'depth_fwd': depth_fwd,
        'depth_bwd': depth_bwd,
    }

    return mip_dict


def rotate_img(img, rotation_angles):
    # ortho_vps = [[0, 0], [-90, 0], [0, -90]]
    rx, rz = rotation_angles
    if rx == 0 and rz == 0:
        return img
    else:
        A = affine_utils.get_affine_matrix_tio_convention(img.shape, {'degrees': (rx, 0, rz)})
        img_rotated = affine_utils.resample_image(
            torch.tensor(img).unsqueeze(0).unsqueeze(0).float(), 
            A, interpolation='bilinear').squeeze(0).squeeze(0).numpy().astype('float32')
        return img_rotated


def projection_pipeline(img_12bit: np.ndarray, 
                        vertebrae_mask_anduin: np.ndarray, 
                        crop_center: np.ndarray, 
                        rotation_angles: Tuple) -> Dict :
    """
    Sumarizes the entire pipeline for generating the MIP projections from the original arterial-phase CT images.

    Args:
        img_12bit (np.ndarray): Raw input image, arterial-phase CT.
        vertebrae_mask_anduin (np.ndarray): Vertebrae mask from anduin.
        crop_center (np.ndarray): The ostium of the SMA in pixel space.
        rotation_angles (Tuple): The rotation angles about the x and z directions.

    Returns:
        Dict: The MIP projections and depth volumes. 
    """    
    # 1. Input: raw image and vertex mask from anduin 

    # 2. Crop both to 300, 300, 150
    img_12bit = crop_image_around_sma_ostium(img_12bit, crop_center)
    vertebrae_mask_anduin = crop_image_around_sma_ostium(vertebrae_mask_anduin, crop_center)

    # 3. Thresholding, rib and vertebrae extraction
    img = preprocess_img(img_12bit, vertebrae_mask_anduin)

    # 4. Crop to 256, 256, 128
    img = center_crop_tensor(img, np.array((256, 256, 128)))

    # 5. Rotate image according to the rotation angles
    rotated_img = rotate_img(img, rotation_angles)

    # 6. Project rotated image along the y axis.
    mip_dict = project_img_bidirectional(rotated_img)
    
    return mip_dict