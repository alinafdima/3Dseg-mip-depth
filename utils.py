import itertools
import logging
import random
from pathlib import Path
from os.path import join

import nibabel as nib
import numpy as np
import torch
from typing import List, Dict

log = logging.getLogger(__name__)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def merge_dictionaries(dict_list: List[Dict], strict=True) -> Dict:
    """
    Merges a list of dictionaries into a single dictionary.
    # It does not check for repeated keys.

    Args:
        dict_list (List[Dict]): _description_

    Returns:
        Dict: A dictionary which merges all the dictionaries in the list.
    """
    merged_dict = {}
    for d in dict_list:
        merged_dict.update(d)

    total_items = sum([len(d) for d in dict_list])
    final_items = len(merged_dict)
    if total_items != final_items:
        all_keys = {[d.keys() for d in dict_list]}
        if strict:
            raise ValueError(f"There are repeated keys in the dictionaries {all_keys}.")
        else:
            log.warning(f"There are repeated keys in the dictionaries {all_keys}.")

    return merged_dict


def save_niftis(save_dict, output_folder):
    Path(output_folder).mkdir(parents=False, exist_ok=True)
    for k, v in save_dict.items():
        nib.save(
            nib.Nifti1Image(v.astype("uint8"), np.eye(4)),
            join(output_folder, f"{k}.nii.gz"),
        )


def save_3d_image(img, save_path, type_conversion="uint8", verbose=False):
    # Convert to an ITKSnap compatible precision
    if type_conversion is not None:
        img = img.astype(type_conversion)

    nib.save(nib.Nifti1Image(img, np.eye(4)), save_path)
    if verbose:
        print("Saved image to: ", save_path)


def read_nifti(data_path):
    return nib.load(data_path).get_fdata()


def flatten_list(l):
    return list(itertools.chain.from_iterable(l))


def preprint_img(im):
    # Takes a 3D slice, and flips the two remaining axes and inverts the
    # y axis for anatomy-faitthful display with matplotlib
    if type(im).__module__ == torch.__name__:
        im = im.numpy()
    img = np.flip(im.transpose(), axis=0)
    return img


def unpreprint_img(img):
    # Reverses the operation of preprint_img
    return np.flip(img, axis=0).transpose()
