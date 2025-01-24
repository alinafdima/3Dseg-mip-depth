import os
from itertools import product
from os.path import join
import h5py
import numpy as np

DATA_ROOT = os.environ["DATA_ROOT"]
HDF5_ROOT = join(DATA_ROOT, "dataset_spectral_CT", "data", "processed")


hdf5_files = {
    # Dict of local hdf5 files
}

depth_files = {
    # Dict of local hdf5 file with depth information
}


class DecoderSpectralCT(object):
    """
    Decodes an hdf5 file encoding data from the Spectral CT dataset
    Closes the hdf5 reader after every access
    """

    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, "r") as data_record:
            self.scan_ids = [x.decode() for x in data_record["scan_ids"]]

    def get_record(self, scan_id, key):
        with h5py.File(self.hdf5_file, "r") as data_record:
            scans = data_record["data"]
            return scans[scan_id][key][:]


class DecoderSpectralCT2(object):
    """
    Decodes an hdf5 file encoding data from the Spectral CT dataset
    Keeps the hdf5 reader the entire time
    """

    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        data_record = h5py.File(self.hdf5_file, "r")
        self.scan_ids = [x.decode() for x in data_record["scan_ids"]]
        self.scans = data_record["data"]

    def get_record(self, scan_id, key):
        return self.scans[scan_id][key][:]

    def close_record(self):
        del self.scans


def decode_entire_spectralCT_dataset(hdf5_file):
    """
    Reads the entire HDF5 record and returns a dictionary with all of the data.
    It only makes sense to do it if the entire dataset is needed

    Args:
        hdf5_file (str): The location of the hdf5 file

    Returns:
        dict(scan_id, data_dict): A dictionary containing of scan_id and data_dict pairings
    """
    data_dict = {}
    with h5py.File(hdf5_file, "r") as data_record:
        scan_ids = [x.decode() for x in data_record["scan_ids"]]
        for scan_id in scan_ids:
            scan_record = data_record["data"][scan_id]

            data_dict[scan_id] = {
                "arterial": scan_record["arterial"][:],
                "seg": scan_record["seg"][:],
            }
        return data_dict


def decode_entire_projections_dataset(hdf5_file):
    """
    Reads the entire HDF5 record and returns a dictionary with all of the data.
    It only makes sense to do it if the entire dataset is needed

    Args:
        hdf5_file (str): The location of the hdf5 file

    Returns:
        dict(scan_id, data_dict): A dictionary containing of scan_id and data_dict pairings
    """
    data_dict = {}
    with h5py.File(hdf5_file, "r") as data_record:
        scan_ids = [x.decode() for x in data_record["scan_ids"]]
        for scan_id in scan_ids:
            scan_record = data_record["data"][scan_id]

            data_dict[scan_id] = {}
            for idx in range(1, 50):
                # data_dict[scan_id][f'projection_{idx}'] = scan_record[f'{idx}'][:]
                data_dict[scan_id][idx] = scan_record[f"{idx}"][:]

        return data_dict


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Canonical and non-canonical projections
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

serialize_viewpoint = lambda rx, rz: f"{int(rx)}_{int(rz)}"
all_projection_angles = np.array(list(product(range(-90, 61, 30), range(-90, 61, 30))))
orthogonal_viewpoints = [[0, 0], [-90, 0], [0, -90]]


def get_available_data_keys(hdf5_file):
    with h5py.File(hdf5_file, "r") as data_record:
        data = data_record["data"]
        scan_record = data[list(data.keys())[0]]
        return list(scan_record.keys())


def get_scan_ids(hdf5_file):
    with h5py.File(hdf5_file, "r") as data_record:
        return [x.decode() for x in data_record["scan_ids"]]


def _validate_scan_ids_list(scan_ids, all_scan_ids):
    if scan_ids is None:
        return all_scan_ids
    if not isinstance(scan_ids, list):
        return [scan_ids]
    else:
        return scan_ids


def load_gt_projections(hdf5_file: str, scan_ids, keys_list) -> dict:
    data_dict = {}
    with h5py.File(hdf5_file, "r") as data_record:
        data = data_record["data"]
        if scan_ids is None:
            scan_ids = [x.decode() for x in data_record["scan_ids"]]
        if keys_list is None:
            keys_list = list(data[list(data.keys())[0]].keys())

        for scan_id in scan_ids:
            data_dict[scan_id] = {}
            scan_record = data[scan_id]
            for key in keys_list:
                data_dict[scan_id][key] = scan_record[key][:]
    return data_dict


def load_skeletons(hdf5_file: str, scan_ids) -> dict:
    all_scan_ids = get_scan_ids(hdf5_file)
    with h5py.File(hdf5_file, "r") as data_record:
        data = data_record["data"]
        scan_ids = _validate_scan_ids_list(scan_ids, all_scan_ids)
        return {scan_id: data[scan_id]["skeleton"][:] for scan_id in scan_ids}


def load_depth(hdf5_file: str, scan_ids, vp_list) -> dict:
    all_scan_ids = get_scan_ids(hdf5_file)
    all_vps = get_available_data_keys(hdf5_file)
    data_dict = {}
    with h5py.File(hdf5_file, "r") as data_record:
        data = data_record["data"]
        scan_ids = _validate_scan_ids_list(scan_ids, all_scan_ids)
        vp_list = _validate_scan_ids_list(vp_list, all_vps)

        for scan_id in scan_ids:
            data_dict[scan_id] = {}
            scan_record = data[scan_id]
            for vp in vp_list:
                data_dict[scan_id][vp] = scan_record[vp][:]
    return data_dict
