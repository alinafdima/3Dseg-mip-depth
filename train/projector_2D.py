# ------------------------------------------------------
# File: projector_2D.py
# Author: Alina Dima <alina.dima@tum.de>
# Created on Thu Feb 02 2023
#
# Projecting 3D volumes to 2D images
# - project_batch() supports different affine matrices for each image in the batch
# - the affine matrices are computed externally, using the tio convention
# ------------------------------------------------------

import torch
import torch.nn.functional as F


def get_multi_channel_grid(affine_matrices: torch.tensor, image_shape) -> torch.Tensor:
    """
    Creates a sampling grid for a single images and multiple affine matrices

    Args:
        affine_matrices (torch.tensor): tensor of shape [N, 3, 4]
        image_shape (list): Image shape (x, y, z)

    Returns:
        torch.Tenosr: Affine grid for resampling an image of the given shape
    """
    assert len(image_shape) == 3, "Only 3D images are supported"
    N = affine_matrices.shape[0]
    image_fullsize = torch.Size((N, 1, *image_shape))

    grid = F.affine_grid(affine_matrices, image_fullsize, align_corners=True)
    return grid


def project_2D(image, projection_axis="y"):
    """
    Projects a channels-first 3D tensor onto a 2D plane.
    It accepts 5D (B, C, x, y, z), 4D (B/C, x, y, z) or 3D (x, y, z) input tensors.

    Args:
        image (torch.tensor): channels-last input tensor
        projection_axis (str): String (x, y, or z) indicating the axis to project onto

    Raises:
        ValueError: If the input tensor is not 3D, 4D, or 5D

    Returns:
        torch.tensor: 2D tensor of shape (x, y) or (x, z) or (y, z), depending on the projection axis, and corresponding number of channels to the input tensor
    """
    if len(image.shape) not in [3, 4, 5]:
        raise ValueError("Only 3D, 4D, and 5D tensors are supported")
    axis_mapping = dict(x=-3, y=-2, z=-1)
    return torch.max(image, axis=axis_mapping[projection_axis])[0]


def project_batch(batch_tensor, affine_matrices_list, image_shape, interpolation):
    """
    Rotates and projects a batch tensor by breaking it up into individual images and reassembling it.
    1. image tensor is obtained by selecting a single image from the batch tensor -> (1, 1, x, y, z)
    2. the image tensor is rotated according to the sampling grid -> (Ch, 1, x, y, z)
    3. the tensor is projected onto 2D -> (Ch, 1, x, y, z) or (Ch, 1, x, z)
    4. the tensor is reshaped into 4-channel 2D tensor: (1, Ch, x, z)
    5. all individual image tensors are concatenated along the batch channel: (B, Ch, x, z)

    Each image in the batch expects a differnt affine matrix stack, which is passed as a list of affine matrices.

    Args:
        batch_tensor (torch.tensor): Input tensor 5D (B, 1, x, y, z), channels-first
        affine_matrices_list: List of affine matrices (Ch x 3 x 4), one for each image in the batch
        image_shape (tuple): Shape of the image (x, y, z) - [256, 256, 128]
        interpolation: Interpolation mode (nearest, bilinear)

    Returns:
        torch.tensor: projections as a tensor of shape (B, Ch, x, z)
        torch.tensor: resampled tensor before projection
    """
    batch_size = batch_tensor.shape[0]
    assert len(affine_matrices_list) == batch_size, "Batch size mismatch"

    resampled_tensors = []
    batch_projections = []
    for img, affine_matrices in zip(batch_tensor, affine_matrices_list):
        projection_channels = affine_matrices.shape[0]
        tensor_resampled = F.grid_sample(
            input=torch.concat([img.unsqueeze(0)] * projection_channels, dim=0),
            grid=get_multi_channel_grid(affine_matrices.cuda(), image_shape),
            mode=interpolation,
            align_corners=True,
        )

        # (Ch, 1, x, z) -> (1, Ch, 1, x, z) -> (1, Ch, x, z)
        tensor_projected = project_2D(tensor_resampled).unsqueeze(0).squeeze(2)
        batch_projections.append(tensor_projected)
        resampled_tensors.append(tensor_resampled.squeeze(1).unsqueeze(0))

    return torch.concat(batch_projections, dim=0), torch.concat(
        resampled_tensors, dim=0
    )
