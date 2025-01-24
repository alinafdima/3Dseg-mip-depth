import logging
import cc3d
import numpy as np
import surface_distance  # https://github.com/deepmind/surface-distance/

log = logging.getLogger(__name__)

# #############################################################################
#                                       Dice
# #############################################################################

EPS = 1e-8


def _oneclass_dice(seg_gt, seg_pred):
    dice = (
        2
        * np.logical_and(seg_gt, seg_pred).sum()
        / (seg_gt.sum() + seg_pred.sum() + EPS)
    )
    return dice


# Includes background
def dice_score(seg_gt, seg_pred, num_classes):
    dice = np.mean(
        [_oneclass_dice(seg_gt == i, seg_pred == i) for i in range(num_classes)]
    )
    return dice


# Foreground only
def foreground_dice_score(seg_gt, seg_pred, num_classes):
    dice = np.mean(
        [_oneclass_dice(seg_gt == i, seg_pred == i) for i in range(1, num_classes + 1)]
    )
    return dice


# #############################################################################
#                           Other TP/FP/FN-based metrics
# #############################################################################


def compute_confusion_matrix(y_true, y_pred):
    # return confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    tp = np.sum(np.logical_and(y_true, y_pred))
    fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
    fn = np.sum(np.logical_and(y_true, np.logical_not(y_pred)))
    tn = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)))

    return np.array([[tp, fp], [fn, tn]])


def compute_balanced_accuracy_from_predictions(y_true, y_pred):
    # return balanced_accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)
    return compute_balanced_accuracy_from_cm(confusion_matrix)


def compute_balanced_accuracy_from_cm(confusion_matrix):
    [tp, fp], [fn, tn] = confusion_matrix
    sensitivity = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)
    return (sensitivity + specificity) / 2


def compute_precision_recall_from_predictions(y_true, y_pred):
    return compute_precision_recall_from_cm(compute_confusion_matrix(y_true, y_pred))


def compute_precision_recall_from_cm(confusion_matrix):
    [tp, fp], [fn, tn] = confusion_matrix
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    return precision, recall


# #############################################################################
#                           Connected components
# #############################################################################


def compute_connected_components(mask, connectivity=26):
    cc = cc3d.connected_components(mask, connectivity=connectivity)

    # Get a list of labels and their frequency, excluding background
    label_list = [
        (x, np.sum((cc == x).astype(int)))
        for x in np.unique(cc)
        if np.sum(cc[cc == x]) > 0
    ]

    return cc, label_list, len(label_list)


def keep_largest_cc(mask, connectivity=26):
    cc, label_list, _ = compute_connected_components(mask, connectivity=connectivity)

    largest_cc, _ = max(label_list, key=lambda x: x[1])
    new_mask = mask * (cc == largest_cc)
    return new_mask


def remove_small_cc(mask, min_size=100, connectivity=26):
    cc, label_list, _ = compute_connected_components(mask, connectivity=connectivity)

    for label, size in label_list:
        if size < min_size:
            mask[cc == label] = 0

    return mask


# #############################################################################
#                           Surface distance-based metrics
# #############################################################################


def compute_distances(mask_gt, mask_y, spacing=(1.0, 1.0, 1.0)):
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_y, spacing
    )
    return surface_distances


def compute_robust_hausdorff_distance(surface_distances, percentile=95):
    hausdorff = surface_distance.compute_robust_hausdorff(surface_distances, percentile)
    return hausdorff


def compute_mean_surface_distance(surface_distances):
    msd_gt_to_pred, msd_pred_to_gt = surface_distance.compute_average_surface_distance(
        surface_distances
    )
    return msd_gt_to_pred, msd_pred_to_gt


def compute_surface_distances_combined(
    mask_gt, mask_y, spacing=(1.0, 1.0, 1.0), percentile=95
):
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_y, spacing
    )
    hausdorff = surface_distance.compute_robust_hausdorff(surface_distances, percentile)
    msd_gt_to_pred, msd_pred_to_gt = surface_distance.compute_average_surface_distance(
        surface_distances
    )

    return hausdorff, msd_gt_to_pred, msd_pred_to_gt
