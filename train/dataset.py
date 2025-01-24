# ------------------------------------------------------
# File: dataset.py
# Author: Alina Dima <alina.dima@tum.de>
# Created on Fri Nov 25 2022
#
# Data loader
#
# ------------------------------------------------------

import logging
import random
import os
from os.path import join
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import Dataset
from tqdm import tqdm

import hdf5_decoder as decoder
from preprocessing.preprocess_images_for_projection import center_crop_tensor
from affine_utils import resample_image, get_affine_matrix_tio_convention

log = logging.getLogger(__name__)

input_shape = (300, 300, 150)
crop_size = (256, 256, 128)
crop_size_4ch = np.array((1, 256, 256, 128))

all_viewpoints = decoder.all_projection_angles
serialize_viewpoint = decoder.serialize_viewpoint
np_to_torch4ch = lambda x: torch.tensor(np.expand_dims(x, axis=0))
crop = lambda x: center_crop_tensor(x, crop_size_4ch)

cv_viewpoints = {}
"""
CV viewpoints are randomly assigned sets of viewpoints for each image sample.
They are meant to simulate scenarios where a specific set of viewpoints 
were manually annotated for each sample. 

It would look something like this:
cv_viewpoints["keyword"] = {
    "subject001": [[-90, -60], [-60, 0]],
    "subject002": [[60, 60], [-90, -60]],
    "subject003": [[-90, -30], [-60, 0]],
    "subject004": [[30, 30], [-90, -60]],
}

For all experiments shown in the paper, we only considered orthogonal viewpoints
orthogonal_viewpoints = [[0, 0], [-90, 0], [0, -90]]

We also experimented with more projection angles, but they weren't beneficial in our experiments
all_projection_angles = np.array(list(product(range(-90, 61, 30), range(-90, 61, 30))))
"""


def get_affine_matrices(selected_angles, crop_size):
    # Create projection affine matrices
    affine_matrices = []
    for rx, rz in selected_angles:
        A = get_affine_matrix_tio_convention(crop_size, {"degrees": (rx, 0, rz)})
        affine_matrices.append(A)
    return torch.concat(affine_matrices, dim=0)


def sample_affine_transformation(A: Dict):
    return {
        "scales": tuple(random.uniform(*A["scales"]) for _ in range(3)),
        "degrees": tuple(random.uniform(*A["degrees"]) for _ in range(3)),
        "translation": tuple(random.uniform(-t, t) for t in A["translation"]),
    }


def get_cv_scan_ids(cv_file, split):
    # Cross-validation
    cv_folder = join(os.environ["REPOS_ROOT"], "ls_segmentation", "cross_validation")
    df = pd.read_csv(join(cv_folder, cv_file), skipinitialspace=True)
    df = df.loc[df.subset == split].reset_index(drop=True)
    scan_ids = [df.scan_id[idx] for idx in range(len(df))]
    return scan_ids


def apply_tio_transforms(img, seg, transform):
    img = tio.ScalarImage(tensor=np_to_torch4ch(img))
    seg = tio.LabelMap(tensor=np_to_torch4ch(seg))
    tio_subject = transform(tio.Subject(img=img, label=seg))
    return tio_subject.img.data.float(), tio_subject.label.data.float()


class InferenceDataset(Dataset):
    def __init__(self, cv_file, split):
        self.data_decoder = decoder.DecoderSpectralCT(
            decoder.hdf5_files["cropped_8bit"]
        )
        self.preprocess = tio.RescaleIntensity(out_min_max=(0, 1))
        self.scan_ids = get_cv_scan_ids(cv_file, split)

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx: int) -> Dict:
        scan_id = self.scan_ids[idx]

        # Read input data
        img, seg = [
            self.data_decoder.get_record(scan_id, key) for key in ["arterial", "seg"]
        ]
        img, seg = apply_tio_transforms(img, seg, self.preprocess)

        return {"scan_id": scan_id, "gt_3d": crop(seg), "input": crop(img)}

    def collate_function(self, input_dictionaries: Dict):
        collated_dict = {}
        for name in ["input", "gt_3d"]:
            collated_dict[name] = torch.stack([x[name] for x in input_dictionaries])
        for name in ["scan_id"]:
            collated_dict[name] = [x[name] for x in input_dictionaries]

        return collated_dict


class DepthDataset(Dataset):
    """
    Dataset for the depth dataset
    """

    def __init__(
        self,
        cv_file: str,
        data_augmentation: bool,
        split: str,
        depth_file: str,
        viewpoint_assignment: str,
        projection_viewpoints: List,
        debug=False,
    ):
        """
        Initializes the dataset.

        Args:
            cv_file (str): disk location of the cross validation file, which is a csv file containing a list of scan_ids and their corresponding fold (train/val/test)
            data_augmentation (bool): Whether to use data augmentation or not
            split (str): One of 'train', 'val', 'test', which is used to select the scan_ids from the cross validation file
            depth_file (str): A depth file form decoder.hdf5_decoder
            viewpoint_assignment(str): either 'fixed' or a keyword in cv_viewpoints
            projection_viewpoints (List): list of projection viewpoints (in case they are fixed)
            debug (bool, optional): Used for debugging. Defaults to False.

        Raises:
            ValueError: If the split is not one of 'train', 'val', 'test'
            ValueError: If the viewpoint assignment is not one of 'fixed' or a keyword in cv_viewpoints
            ValueError: If the viewpoint assignment is 'fixed' and the projection viewpoints are not provided
            ValueError: If the viewpoint assignment is not 'fixed' and the projection viewpoints are provided
        """
        ###########################################
        # Input validation
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        if viewpoint_assignment not in ["fixed"] + list(cv_viewpoints.keys()):
            raise ValueError(f"Invalid viewpoint assignment: {viewpoint_assignment}")
        if viewpoint_assignment == "fixed" and projection_viewpoints is None:
            raise ValueError(
                "Fixed viewpoint assignment requires a list of projection viewpoints"
            )
        if viewpoint_assignment != "fixed" and projection_viewpoints is not None:
            raise ValueError(
                f"Experted projection_viewpoints to be None for viewpoint assignment {viewpoint_assignment}"
            )

        ###########################################
        # Save parameters
        self.subset = split
        self.data_augmentation = data_augmentation
        self.debug = debug

        ###########################################
        # Data augmentation
        self.affine_transform = {
            "scales": (0.8, 1.2),
            "degrees": (-15, 15),
            "translation": (30, 30, 3),
        }
        self.affine_probability = 0.9
        self.intensity_transforms = [
            tio.RandomBlur(std=(0, 2), exclude=("label",), p=0.75),
            tio.RandomGamma(log_gamma=(-0.3, 0.3), exclude=("label",), p=0.75),
        ]

        ###########################################
        # Preprocessing
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        if self.data_augmentation:
            self.preprocess = tio.Compose(
                [rescale] + self.intensity_transforms + [rescale]
            )
        else:
            self.preprocess = rescale

        ###########################################
        # Cross-validation
        self.scan_ids = get_cv_scan_ids(cv_file, split)

        ###########################################
        # Viewpoint assignment
        self.viewpoint_strategy = viewpoint_assignment
        if self.viewpoint_strategy == "fixed":
            self.viewpoints = {
                scan_id: projection_viewpoints for scan_id in self.scan_ids
            }
        else:
            self.viewpoints = {
                scan_id: cv_viewpoints[viewpoint_assignment][scan_id]
                for scan_id in self.scan_ids
            }

        self.viewpoints_serialized = {}
        self.A_projections = {}
        for scan_id in self.scan_ids:
            vps = self.viewpoints[scan_id]
            self.viewpoints_serialized[scan_id] = [
                serialize_viewpoint(rx, rz) for rx, rz in vps
            ]
            self.A_projections[scan_id] = get_affine_matrices(vps, crop_size)

        ###########################################
        # Load data / paths

        # Depth
        self.depth_file = decoder.depth_files[depth_file]

        # Images and 3D GT
        self.data_decoder = decoder.DecoderSpectralCT(
            decoder.hdf5_files["cropped_8bit"]
        )

        # Read projections
        log.info(f"Loading ground truth projections for {len(self.scan_ids)} scans...")

        self.gt_projections = {}
        for scan_id in tqdm(self.scan_ids):
            viewpoints = self.viewpoints_serialized[scan_id]
            gt_projections = decoder.load_gt_projections(
                hdf5_file=decoder.hdf5_files["projections_non_canonical"],
                scan_ids=[scan_id],
                keys_list=viewpoints,
            )[scan_id]
            projections_list = [torch.tensor(gt_projections[x]) for x in viewpoints]
            self.gt_projections[scan_id] = torch.stack(projections_list, dim=0)

        log.info(f"Dataset '{split}' initialized")

    def __len__(self):
        return len(self.scan_ids)

    def spatial_augmentation(self):
        if self.data_augmentation:
            p = self.affine_probability
            return random.choices([True, False], weights=[p, 1 - p])[0]
        else:
            return False

    def __getitem__(self, idx: int) -> Dict:
        scan_id = self.scan_ids[idx]

        # Read input data
        img, seg = [
            self.data_decoder.get_record(scan_id, key) for key in ["arterial", "seg"]
        ]
        img, seg = apply_tio_transforms(img, seg, self.preprocess)

        # Spatial augmentation (if needed)
        if self.spatial_augmentation():
            affine_params = sample_affine_transformation(self.affine_transform)
            A_forward = get_affine_matrix_tio_convention(input_shape, affine_params)
            img_affine = resample_image(
                img.unsqueeze(0), A_forward, interpolation="bilinear"
            )

            sample_dict = {
                "input": crop(img_affine.squeeze(0)),
                "input_nonaffine": crop(img),
                "affine_params": affine_params,
            }
        else:
            sample_dict = {"input": crop(img)}

        # Read depth 3D map from disk
        viewpoints = self.viewpoints_serialized[scan_id]
        depth_npy = decoder.load_depth(self.depth_file, scan_id, viewpoints)[scan_id]
        depth = torch.concat([np_to_torch4ch(depth_npy[x]) for x in viewpoints], dim=0)

        return {
            **sample_dict,
            "scan_id": scan_id,
            "gt_3d": crop(seg),
            "A_projections": self.A_projections[scan_id],
            "gt_projections": self.gt_projections[scan_id],
            "viewpoints": self.viewpoints[scan_id],
            "viewpoints_serialized": self.viewpoints_serialized[scan_id],
            "depth": depth,
        }

    def collate_function(self, input_dictionaries: Dict):
        collated_dict = {}
        for name in ["input", "gt_3d", "gt_projections", "depth"]:
            collated_dict[name] = torch.stack([x[name] for x in input_dictionaries])
        for name in ["scan_id", "A_projections", "viewpoints", "viewpoints_serialized"]:
            collated_dict[name] = [x[name] for x in input_dictionaries]
        for name in ["input_nonaffine", "affine_params"]:
            collated_dict[name] = [
                x[name] if name in x else None for x in input_dictionaries
            ]

        return collated_dict


if __name__ == "__main__":
    print("-----> call to dataset.py")

    logging.basicConfig(level=logging.INFO)
    dataset = DepthDataset(
        cv_file="final_cv1.csv",
        depth_file="depth_v1_ortho",
        viewpoint_assignment="ortho1_v1",
        projection_viewpoints=None,
        split="train",
        data_augmentation=False,
    )
    first_element = dataset[0]
    print(first_element.keys())
