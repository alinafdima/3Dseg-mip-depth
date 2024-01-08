# ------------------------------------------------------
# File: affine_utils.py
# Author: Alina Dima <alina.dima@tum.de>
#
# Utils for data augmentation
# Generate a sampling grid
# ------------------------------------------------------

import torch
import torch.nn.functional as F
from pytorch3d.transforms import euler_angles_to_matrix


def get_affine_matrix_tio_convention(image_shape, transform) -> torch.Tensor:
    """
    Returns the affine matrix for a given transform dictionary
    Translation, rotation and scaling are all according to the torchio convention

    Args:
        image_shape (list): [size x, size y, size z]
        transform (dict): dictionary with keys: translation, degrees, scales

    Returns:
        torch.Tensor: 4x4 affine matrix (R|t)
    """
    assert len(image_shape) == 3, 'Only 3D images are supported'
    acceptable_keys = ['translation', 'degrees', 'scales']
    provided_keys = list(transform.keys())
    assert all([k in acceptable_keys for k in provided_keys]), 'Unkown keys in transform: ' + str(provided_keys)

    # Populate missing keys
    if 'translation' not in transform:
        transform['translation'] = [0, 0, 0]
    if 'degrees' not in transform:
        transform['degrees'] = [0, 0, 0]
    if 'scales' not in transform:
        transform['scales'] = [1, 1, 1]

    convention = 'YZX'
    translation = torch.tensor(transform['translation']).flip(0).float()
    rotation_angles = torch.deg2rad(torch.tensor(transform['degrees']).flip(0)).float()
    scale_factors = torch.tensor(transform['scales']).flip(0).float()
    image_shape = torch.tensor(image_shape).flip(0)

    # Translation
    translation_original_coord_system = (translation  / image_shape * -2).reshape(3, -1)
    
    # Rotation
    rotation_map = dict(zip('XYZ', rotation_angles))
    rotation_matrix = euler_angles_to_matrix(torch.tensor(
        [rotation_map[convention[x]] for x in range(3)]), convention=convention)
    
    # Scaling
    rescale_matrix = torch.diag(1./ scale_factors)
    
    # Make coordinate system isotropic (since voxel size varies in the 3 dimensions)
    size_scales = image_shape / torch.max(image_shape)
    matrix_isotropic = torch.diag(1. / size_scales)
    matrix_anisotropic = torch.diag(size_scales)

    rotation_matrix_rescaled = torch.linalg.multi_dot([
        matrix_isotropic,
        rotation_matrix, 
        rescale_matrix,
        matrix_anisotropic,
        ])

    coordinate_change = torch.linalg.multi_dot([
        matrix_isotropic,
        rotation_matrix, 
        matrix_anisotropic,
        ])

    affine_matrix = torch.concat((
        rotation_matrix_rescaled, 
        torch.matmul(coordinate_change, translation_original_coord_system)
        ), axis=1).unsqueeze(0)

    return affine_matrix


def get_inverse_affine_matrix_tio_convention(image_shape, transform):
    assert len(image_shape) == 3, 'Only 3D images are supported'
    acceptable_keys = ['translation', 'degrees', 'scales']
    provided_keys = list(transform.keys())
    assert all([k in acceptable_keys for k in provided_keys]), 'Unkown keys in transform: ' + str(provided_keys)

    # Populate missing keys
    if 'translation' not in transform:
        transform['translation'] = [0, 0, 0]
    if 'degrees' not in transform:
        transform['degrees'] = [0, 0, 0]
    if 'scales' not in transform:
        transform['scales'] = [1, 1, 1]

    convention = 'YZX'
    translation = torch.tensor(transform['translation']).flip(0).float()
    rotation_angles = torch.deg2rad(torch.tensor(transform['degrees']).flip(0)).float()
    scale_factors = torch.tensor(transform['scales']).flip(0).float()
    image_shape = torch.tensor(image_shape).flip(0)

    # Translation
    translation_original_coord_system = (translation  / image_shape * 2).reshape(3, -1)
    
    # Rotation
    rotation_map = dict(zip('XYZ', rotation_angles))
    rotation_matrix = torch.linalg.inv(euler_angles_to_matrix(torch.tensor(
        [rotation_map[convention[x]] for x in range(3)]), convention=convention))
    
    # Scaling
    rescale_matrix = torch.diag(scale_factors)
    
    # Make coordinate system isotropic (since voxel size varies in the 3 dimensions)
    size_scales = image_shape / torch.max(image_shape)
    matrix_isotropic = torch.diag(1. / size_scales)
    matrix_anisotropic = torch.diag(size_scales)

    rotation_matrix_rescaled = torch.linalg.multi_dot([
        matrix_isotropic,
        rescale_matrix,
        rotation_matrix,
        matrix_anisotropic,
        ])

    coordinate_change = torch.linalg.multi_dot([
        matrix_isotropic,
        rescale_matrix,
        matrix_anisotropic,
        ])

    affine_matrix = torch.concat((
        rotation_matrix_rescaled, 
        torch.matmul(coordinate_change, translation_original_coord_system)
        ), axis=1).unsqueeze(0)

    return affine_matrix


def resample_image(img, affine_matrix, interpolation='bilinear', requires_grad=False, cuda=False):
    assert len(img.shape) == 5, 'Only 5D images are supported'
    sampling_grid = F.affine_grid(affine_matrix, img.size(), align_corners=True)
    sampling_grid.requires_grad = requires_grad
    if cuda:
        sampling_grid = sampling_grid.cuda()
    return F.grid_sample(img, sampling_grid, mode=interpolation, align_corners=True)


def chain_canonical_affine_mat(A_canonical, A_affine):
    R_canonical = A_canonical[:, :3, :3]
    A_chained = torch.matmul(R_canonical, A_affine)
    return A_chained
