import logging
import os
from os.path import join
from pathlib import Path
from itertools import chain, product
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes

import metrics
import visualization as vis
from utils import merge_dictionaries, save_niftis
import hdf5_decoder as decoder

log = logging.getLogger(__name__)


# ############################################################
# Aggregator for computing batch-wise best/worst metric values
# ############################################################


class AggregatorExtremeValues:
    """
    Class for aggregating the extreme (best/worst) metric values
    (eg. for saving the best validation model during training or visualizing the worst predictions of an epoch).

    Usage:
        1. Initialize aggregator with the desired metric name and whether higher or lower is better.
        eg. for the loss the best value is the lowest, and for dice_2d the best value is the highest.
        evaluator = AggregatorExtremeValues({'loss': False, 'dice_2d': True})

        2. For each batch, add the metric values and their corresponding ids to the aggregator.
        for batch_idx in range(batches):
            evaluator.update(batch_values, batch_ids)

        3. Retrieve the extreme values at the end of an epoch.
        evaluator.optimal_values

    """

    def __init__(self, metrics_dict, keep_only_best=False):
        if keep_only_best:
            self.checkpoint_metrics = {
                f"{metric}_best": maximize for metric, maximize in metrics_dict.items()
            }
        else:
            self.checkpoint_metrics = merge_dictionaries(
                [
                    {
                        f"{metric}_best": maximize
                        for metric, maximize in metrics_dict.items()
                    },
                    {
                        f"{metric}_worst": not maximize
                        for metric, maximize in metrics_dict.items()
                    },
                ]
            )

        init_value = lambda maximize: -np.inf if maximize else np.inf
        self.optimal_values = {
            metric: (init_value(maximize), None)
            for metric, maximize in self.checkpoint_metrics.items()
        }

    def log_batch(
        self, batch_metrics: Dict[str, List], batch_ids: List[str], debug=False
    ) -> Dict:
        """
        Updates the best/worst values for each metric with a batch of values

        Args:
            batch_metrics (Dict): A dictionary containing the batch values for each metric
            batch_ids (List[str]): The scan ids of the current batch
        """

        dict_updated = {}
        for key, maximize in self.checkpoint_metrics.items():
            metric = "_".join(key.split("_")[:-1])
            self.optimal_values[key], is_updated = self._update_value(
                maximize=maximize,
                batch_values=zip(batch_metrics[metric], batch_ids),
                current_optimal_tuple=self.optimal_values[key],
            )
            if is_updated:
                dict_updated[key] = self.optimal_values[key]

        return dict_updated

    def log_single_value(self, metrics_dict, entry_id, debug=False) -> Dict:
        """
        Updates the best/worst values for each metric with a single new entry

        Args:
            metrics_dict (Dict): A dictionary of metric values
            entry_id (str): The id of the entry
        """
        return self.log_batch(
            {k: [v] for k, v in metrics_dict.items()}, [entry_id], debug=debug
        )

    @staticmethod
    def _update_value(
        maximize: bool, batch_values, current_optimal_tuple: Tuple[float, str]
    ) -> Tuple[Tuple, bool]:
        """
        Updates the best/worst metric aggregator from a list of batch metric values and their corresponding ids.
        Each batch metric value is compared to the current best/worst metric aggregator value, and the new optimal value is returned.
        Each tuple consists of the metric value and its corresponding id.

        Args:
            maximize (bool): Whether the metric should be minimized or maximized
            batch_values (List[Tuple]): Values from the current batch
            current_optimal_tuple (Tuple[float, str]): Current best/worst item (before the current batch)

        Returns:
            Tuple[float, str]: New optimal value including the current batch
        """
        cmp_fun = max if maximize else min
        batch_tuple = cmp_fun(batch_values, key=lambda x: x[0])

        if cmp_fun(batch_tuple[0], current_optimal_tuple[0]) == batch_tuple[0]:
            return batch_tuple, True
        else:
            return current_optimal_tuple, False


# ############################################################


def compute_metrics(
    metrics_categories: List[str],
    metrics_names: List[str],
    tensors_npy: Dict[str, np.ndarray],
    dimensionality: str,
) -> Dict[str, np.ndarray]:
    """
    Computes all metrics in metrics_names for the given tensors_npy, by calling the corresponding function in metrics.py
    It computes either 2D or 3D metrics.
    Some metrics are bundled up for efficiency reasons (eg. surface-based metrics).

    Args:
        metrics_names (List[str]): The names of the metrics to compute
        tensors_npy (Dict[str, np.ndarray]): The tensors to compute the metrics on
        dimensionality (str): '2d' or '3d'

    Returns:
        Dict[str, np.ndarray]: The metrics, with the same keys as metrics_names and an array of values for each metric,
            where each value corresponds to a scan
    """
    log.debug("Computing metrics")

    gt, pred, seg = [
        tensors_npy[f"{name}_{dimensionality}"] for name in ["gt", "pred", "seg"]
    ]
    B, C = gt.shape[0], gt.shape[1]

    connectivity = 6 if dimensionality == "3d" else 4
    spacing = (1.0, 1.0, 1.0) if dimensionality == "3d" else (1.0, 1.0)
    robust_percentile = 95

    batch_metrics = {key: np.zeros(B) for key in metrics_names}
    for batch_idx in range(B):
        channel_metrics = {key: 0.0 for key in metrics_names}
        for channel_idx in range(C):
            gt_sample = gt[batch_idx, channel_idx]
            pred_sample = pred[batch_idx, channel_idx]
            seg_sample = seg[batch_idx, channel_idx]

            if "dice" in metrics_categories:
                log.debug("Dice")
                # dice = metrics.dice_score(gt_sample, seg_sample, num_classes=2)
                dice = metrics.foreground_dice_score(
                    gt_sample, seg_sample, num_classes=1
                )
                channel_metrics["dice"] += dice

            if "surface_based" in metrics_categories:
                log.debug("Surface distance - based")
                if seg_sample.max() == 0.0:
                    large_value = 1e9
                    for key in ["hausdorff", "msd_gt_to_pred", "msd_pred_to_gt"]:
                        channel_metrics[key] += large_value
                    log.warning(
                        "Segmentation produced only 0, setting surface-based metrics to a large value"
                    )
                elif channel_metrics["dice"] < 0.5:
                    log.warning("Dice is below 0.5, setting surface-based metrics to 0")
                    channel_metrics["hausdorff"] += 0
                    channel_metrics["msd_gt_to_pred"] += 0
                    channel_metrics["msd_pred_to_gt"] += 0
                else:
                    hausdorff, msd_gt_to_pred, msd_pred_to_gt = (
                        metrics.compute_surface_distances_combined(
                            gt_sample.astype(bool),
                            seg_sample.astype(bool),
                            spacing=spacing,
                            percentile=robust_percentile,
                        )
                    )
                    channel_metrics["hausdorff"] += hausdorff
                    channel_metrics["msd_gt_to_pred"] += msd_gt_to_pred
                    channel_metrics["msd_pred_to_gt"] += msd_pred_to_gt

            if "confusion_matrix_based" in metrics_categories:
                log.debug("Confusion matrix - based")
                confusion_matrix = metrics.compute_confusion_matrix(
                    gt_sample, seg_sample
                )
                precision, recall = metrics.compute_precision_recall_from_cm(
                    confusion_matrix
                )
                balanced_accuracy = metrics.compute_balanced_accuracy_from_cm(
                    confusion_matrix
                )
                channel_metrics["precision"] += precision
                channel_metrics["recall"] += recall
                channel_metrics["balanced_accuracy"] += balanced_accuracy

            if "connected_components" in metrics_categories:
                log.debug("Connected components")
                if channel_metrics["dice"] < 0.5:
                    log.warning("Dice is below 0.5, setting surface-based metrics to 0")
                    channel_metrics["connected_components"] += 0
                else:
                    _, _, no_cconnected_components = (
                        metrics.compute_connected_components(
                            seg_sample, connectivity=connectivity
                        )
                    )
                    channel_metrics["connected_components"] += no_cconnected_components

        for key in metrics_names:
            batch_metrics[key][batch_idx] = channel_metrics[key] / C
    log.debug("Metrics computed")
    return batch_metrics


# ############################################################

"""
Dictionary that maps the metrics to whether they should be maximized or minimized

Returns:
    bool: True if higher is better, false if lower is better
"""
metrics_optimization_lookup_table = {
    "loss": False,
    "dice": True,
    "balanced_accuracy": True,
    "precision": True,
    "recall": True,
    "hausdorff": False,
    "msd_gt_to_pred": False,
    "msd_pred_to_gt": False,
    "roc_auc": True,
    "connected_components": False,
}


def _is_maximize(metric):
    if metric.endswith("3d") or metric.endswith("2d"):
        return metrics_optimization_lookup_table[metric[:-3]]
    else:
        return metrics_optimization_lookup_table[metric]


# ############################################################

metrics_categories_lookup_table = {
    "dice": ["dice"],
    "surface_based": ["hausdorff", "msd_gt_to_pred", "msd_pred_to_gt"],
    "confusion_matrix_based": ["precision", "recall", "balanced_accuracy"],
    "roc_auc": ["roc_auc"],
    "connected_components": ["connected_components"],
}


def expand_metrics_categories(metrics_categories: List[str]) -> List[str]:
    return list(
        chain.from_iterable(
            [metrics_categories_lookup_table[x] for x in metrics_categories]
        )
    )


# ############################################################
all_loss_names = ["loss", "loss_depth", "loss_2d", "loss_3d"]


class EvaluatorTraining:
    """
    Evaluator class used during training.
    It tracks all of the metrics for each epoch, as well as the best/worst values for loss and dice needed for model selection.
    """

    def __init__(
        self, metrics_categories, mode, metrics_files, track_ids, output_folder
    ):
        self.metrics_categories = metrics_categories
        self.metrics_names = expand_metrics_categories(metrics_categories)
        self.mode = mode
        self.visualization_metrics = {
            f"{metric}_{dim}": _is_maximize(metric)
            for metric, dim in product(self.metrics_names, ["2d", "3d"])
        }
        self.checkpoint_metrics = {
            metric: _is_maximize(metric) for metric in ["loss", "dice_2d", "dice_3d"]
        }
        self.metrics_files = metrics_files
        self.track_ids = track_ids
        self.output_folder = output_folder
        Path(self.output_folder).mkdir(parents=False, exist_ok=True)

    # Global aggregators
    def init_global(self):
        self.best_models = AggregatorExtremeValues(
            self.checkpoint_metrics, keep_only_best=True
        )

    def update_global(self, epoch_dict, epoch_number):
        dict_updated = self.best_models.log_single_value(
            metrics_dict={
                metric: epoch_dict[metric] for metric in self.checkpoint_metrics
            },
            entry_id=epoch_number,
            debug=True,
        )
        return dict_updated

    # Epoch aggregators
    def init_epoch(self):
        self.aggregator_samples = AggregatorExtremeValues(self.visualization_metrics)
        self.metrics_accumulators = merge_dictionaries(
            [
                # {'loss': [], 'loss_depth': [], 'loss_2d': [], 'loss_3d': []},
                {key: [] for key in all_loss_names},
                {
                    f"{metric}_{dim}": []
                    for metric, dim in product(self.metrics_names, ["2d", "3d"])
                },
            ]
        )
        self.epoch_scan_ids = []
        self.extreme_samples = {
            key: None for key in self.aggregator_samples.optimal_values.keys()
        }

    def log_batch(self, tensors, sample_dict, loss_dict, epoch):
        batch_scan_ids = sample_dict["scan_id"]
        self.epoch_scan_ids.extend(batch_scan_ids)
        for key, value in loss_dict.items():
            self.metrics_accumulators[key].append(value)

        # Compute and collect metrics
        metric_dict_2d = compute_metrics(
            self.metrics_categories, self.metrics_names, tensors, "2d"
        )
        metric_dict_2d = {
            f"{metric}_2d": value for metric, value in metric_dict_2d.items()
        }

        metric_dict_3d = compute_metrics(
            self.metrics_categories, self.metrics_names, tensors, "3d"
        )
        metric_dict_3d = {
            f"{metric}_3d": value for metric, value in metric_dict_3d.items()
        }

        batch_metrics = merge_dictionaries([metric_dict_2d, metric_dict_3d])
        for key, batch_values in batch_metrics.items():
            self.metrics_accumulators[key].extend(batch_values)

        # Check if any of the new samples are best / worst so far
        dict_updated = self.aggregator_samples.log_batch(batch_metrics, batch_scan_ids)

        # For each updated metric, save the corresponding data
        for key, (value, scan_id) in dict_updated.items():
            # Find id corresponding to the scan id
            scan_idx = batch_scan_ids.index(scan_id)
            self.extreme_samples[key] = merge_dictionaries(
                [
                    {
                        "scan_id": scan_id,
                        "value": value,
                    },
                    {key: tensor[scan_idx, ...] for key, tensor in tensors.items()},
                ]
            )

    def finalize_epoch(self, epoch):
        epoch_dict = {
            key: np.mean(values) for key, values in self.metrics_accumulators.items()
        }
        self.log_final_epoch_data(epoch_dict, epoch)
        self.save_epoch_metrics(epoch_dict, epoch)
        return epoch_dict

    def save_epoch_metrics(self, epoch_dict, epoch):
        def append_or_create(df, file_path):
            if not os.path.isfile(file_path):
                df.to_csv(file_path, header="column_names")
            else:
                df.to_csv(file_path, mode="a", header=False)

        # Save average metrics
        df = pd.DataFrame({**epoch_dict, "epoch": epoch}, index=[0])
        append_or_create(df, self.metrics_files["averages"])

        # Save sample-wise metrics
        saved_metrics = {
            key: [f"{val:.4f}" for val in val_list]
            for key, val_list in self.metrics_accumulators.items()
            if not key.startswith("loss")
        }
        dl_size = len(self.epoch_scan_ids)
        df = pd.DataFrame(
            {
                "scan_id": self.epoch_scan_ids,
                "epoch": [epoch] * dl_size,
                **saved_metrics,
            }
        )
        append_or_create(df, self.metrics_files["sample_wise"])

    def log_final_epoch_data(self, epoch_dict, epoch):
        # Wandb logging
        epoch_dict_prefixed = {
            f"{self.mode}_{key}": val for key, val in epoch_dict.items()
        }
        # wandb.log({**epoch_dict_prefixed, "epoch": epoch}, step=epoch)

        # Log file logging
        dice_dict = {
            key: val for key, val in epoch_dict.items() if key.startswith("dice")
        }
        dice_print = " | ".join([f"{k}: {v*100:.2f}" for k, v in dice_dict.items()])
        loss_print = " | ".join([f"{k}: {epoch_dict[k]:.4f}" for k in all_loss_names])
        log.info(f"[{self.mode}] Epoch: {epoch} | {loss_print} | {dice_print}")

        epoch_dict_prefixed_best = {
            f"best_{self.mode}_{key}": val for key, val in epoch_dict.items()
        }
        # wandb.run.summary.update(epoch_dict_prefixed_best)


# #############################################################################################
# Evaluation
# #############################################################################################
pp_3d = lambda x: x[0, 0, ...]
pp_2d = lambda x: x[0, ...]
percent_print = lambda x: f"{float(x)*100:.2f}%"


def hole_filling(seg):
    seg_whole = binary_fill_holes(seg, structure=np.ones((3, 3, 1))).astype(int)
    seg_whole = binary_fill_holes(seg_whole, structure=np.ones((1, 3, 3))).astype(int)
    seg_whole = binary_fill_holes(seg_whole, structure=np.ones((3, 1, 3))).astype(int)
    seg_whole = binary_fill_holes(seg_whole, structure=np.ones((3, 3, 1))).astype(int)
    seg_whole = binary_fill_holes(seg_whole, structure=np.ones((1, 3, 3))).astype(int)
    seg_whole = binary_fill_holes(seg_whole, structure=np.ones((3, 1, 3))).astype(int)
    seg_whole = binary_fill_holes(seg_whole, structure=np.ones((2, 2, 3))).astype(int)
    seg_whole = binary_fill_holes(seg_whole, structure=np.ones((3, 2, 2))).astype(int)
    seg_whole = binary_fill_holes(seg_whole, structure=np.ones((2, 3, 2))).astype(int)
    return seg_whole


class InferenceEvaluator:
    def __init__(self, epoch, data_split, experiment_folder, run):
        self.epoch = epoch
        self.data_split = data_split
        self.experiment_folder = experiment_folder
        self.run = run
        self.output_folder = join(self.experiment_folder, f"inference_{data_split}")
        Path(self.output_folder).mkdir(parents=False, exist_ok=True)

        self.metrics_categories = [
            "dice",
            "confusion_matrix_based",
            "connected_components",
            "surface_based",
        ]
        self.metrics_names = expand_metrics_categories(self.metrics_categories)

        self.metrics_accumulators = merge_dictionaries(
            [
                {
                    f"{metric}_{dim}": []
                    for metric, dim in product(self.metrics_names, ["3d"])
                },
                {f"{metric}": [] for metric in ["skeleton_recall"]},
            ]
        )
        self.scan_ids = []

    def log_batch(self, tensors, sample_dict):
        scan_id = sample_dict["scan_id"][0]

        # Collect metrics
        metric_dict_3d = compute_metrics(
            self.metrics_categories, self.metrics_names, tensors, "3d"
        )
        metric_dict_3d = {
            f"{metric}_3d": value for metric, value in metric_dict_3d.items()
        }

        # batch_metrics = merge_dictionaries([metric_dict_2d, metric_dict_3d])
        batch_metrics = metric_dict_3d
        for key, batch_values in batch_metrics.items():
            self.metrics_accumulators[key].extend(batch_values)

        self.scan_ids.append(scan_id)

        skeleton_gt = decoder.load_skeletons(
            decoder.hdf5_files["skeletons_non_canonical"], scan_id
        )[scan_id]
        seg_pred = pp_3d(tensors["seg_3d"])
        seg_gt = pp_3d(tensors["gt_3d"])
        seg_whole = hole_filling(seg_pred)

        skeleton_recall = np.sum(skeleton_gt * seg_whole) / np.sum(skeleton_gt)
        print(
            f"[{scan_id}] "
            + f"| dice 3d: {percent_print(metric_dict_3d['dice_3d'])}"
            + f"| skeleton recall: {percent_print(skeleton_recall)}"
        )

        self.metrics_accumulators["skeleton_recall"].append(skeleton_recall)

        # Save 3D data
        pred_cm = vis.color_code_predictions_3d(seg_gt, seg_whole)
        skeleton_gt_cm = vis.color_code_skeleton_intersection(skeleton_gt, seg_whole)
        output_folder_3d = join(self.output_folder, "images_3d")
        Path(output_folder_3d).mkdir(parents=False, exist_ok=True)
        save_niftis(
            {
                f"{scan_id}_input": pp_3d(tensors["input"]) * 255,
                f"{scan_id}_gt": seg_gt,
                f"{scan_id}_seg_pred": seg_pred,
                f"{scan_id}_seg_whole": seg_whole,
                f"{scan_id}_seg_cm": pred_cm,
                f"{scan_id}_skeleton_gt": skeleton_gt,
                f"{scan_id}_skeleton_gt_intersections": skeleton_gt_cm,
            },
            output_folder_3d,
        )

    def finalize(self, global_epoch=0):
        epoch_dict = {
            key: np.mean(values) for key, values in self.metrics_accumulators.items()
        }

        # Log file logging
        dice_dict = {
            key: val for key, val in epoch_dict.items() if key.startswith("dice")
        }
        dice_print = " | ".join([f"{k}: {v*100:.2f}" for k, v in dice_dict.items()])
        print(f"Inference [{self.data_split}] Epoch: {self.epoch} | {dice_print}")

        samplewise_metrics = {
            key: [round(val, 5) for val in val_list]
            for key, val_list in self.metrics_accumulators.items()
        }
        wandb_dict = {
            "epoch": self.epoch,
            "metrics": epoch_dict,
            "samplewise_metrics": samplewise_metrics,
            "scan_ids": self.scan_ids,
        }
        if self.run is not None:
            # pprint(wandb_dict)
            self.run.config[f"results_{self.data_split}"] = wandb_dict
            self.run.update()
        else:
            # wandb.config[f"results_{self.data_split}"] = wandb_dict
            epoch_dict_prefixed = {
                f"{self.data_split}_{key}": val for key, val in epoch_dict.items()
            }
            # wandb.log({**epoch_dict_prefixed, "epoch": self.epoch}, step=global_epoch)

        log.info("Updated results in wandb")
        return wandb_dict
