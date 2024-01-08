# ------------------------------------------------------
# File: depth_propagation.py
# Author: Alina Dima <alina.dima@tum.de>
#
# The process of deph map propagation, from a 2D annotation to a 3D depth map
# ------------------------------------------------------


import itertools
import numpy as np
import torch
import torch.nn.functional as F


def construct_depth_volume(annotation_2d, projection_depth, is_inverted, output_crop_size=(256, 256, 128)):
    y_naught = 255 if is_inverted else 0
    depth_volume = np.zeros(output_crop_size)
    sx, sz = annotation_2d.shape
    for x, z in itertools.product(range(sx), range(sz)):
        if annotation_2d[x, z] == 1:
            y = projection_depth[x, z]
            if y != y_naught:
                depth_volume[x, y, z] = 1

    return depth_volume


def merge_depth_volumes(depth_fwd, depth_bwd):
    npy3ch_to_torch5ch = lambda x: torch.tensor(x).unsqueeze(0).unsqueeze(0).float()
    torch5ch_to_npy3ch = lambda x: x.squeeze(0).squeeze(0).numpy()

    speckle_filter1 = np.ones((3, 3, 3))
    speckle_filter1[1, 1, 1] = 0
    speckle_filter1 = npy3ch_to_torch5ch(speckle_filter1)
    speckle_filter2 = np.ones((11, 11, 11))
    speckle_filter2[1, 1, 1] = 0
    speckle_filter2 = npy3ch_to_torch5ch(speckle_filter2)

    vol_npy = np.logical_or(depth_fwd, depth_bwd).astype(int)

    # Remove speckle noise
    vol_torch = npy3ch_to_torch5ch(vol_npy)
    convolution_result1 = torch5ch_to_npy3ch(F.conv3d(vol_torch, speckle_filter1, padding='same', stride=1))
    vol_npy = ((vol_npy * convolution_result1) > 0).astype(int)

    vol_torch = npy3ch_to_torch5ch(vol_npy)
    convolution_result2 = torch5ch_to_npy3ch(F.conv3d(vol_torch, speckle_filter2, padding='same', stride=1))
    vol_npy = ((vol_npy * convolution_result2) > 10).astype(int)

    return vol_npy


def double_depth_propagation(anno, depth_fwd, depth_bwd):
    depth_volume_fwd = construct_depth_volume(anno, depth_fwd, is_inverted=False)
    depth_volume_bwd = construct_depth_volume(anno, depth_bwd, is_inverted=True)
    depth_volume = merge_depth_volumes(depth_volume_fwd, depth_volume_bwd)
    return depth_volume


def fill_depth(img_12bit, depth_volume_hollow, th=0.01):
    depth_volume_partially_filled = np.copy(depth_volume_hollow)
    for x in range(256):
        for z in range(128):
            ray = depth_volume_hollow[x, :, z]
            ray_pixels = np.where(ray == 1)
            if len(ray_pixels[0]) != 2:
                continue
            min_ray = np.min(ray_pixels)
            max_ray = np.max(ray_pixels)
            if min_ray == 0 or max_ray == 255:
                continue
            img_ray = img_12bit[x, min_ray : max_ray, z]
            if img_ray.max() - img_ray.min() > th:
                continue
            depth_volume_partially_filled[x, min_ray:max_ray, z] = 1
    return depth_volume_partially_filled
