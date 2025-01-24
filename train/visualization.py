# ------------------------------------------------------
# File: color_coding_cm.py
# Author: Alina Dima <alina.dima@tum.de>
# Created on Fri Nov 18 2022
#
# ------------------------------------------------------

import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid

from utils import preprint_img, flatten_list


cm_color_coding_2d = {
    "tp": (255, 248, 240),  # floral white, #FFF8F0
    "fn": (160, 108, 213),  # amethyst, #A06CD5
    "fp": (237, 106, 90),  # terra cotta, #ED6A5A
    "tn": (19, 17, 18),  # smoky black, #131112
}

cm_color_coding_3d = {
    "tn": 0,
    "tp": 1,
    "fn": 2,
    "fp": 3,
}


def color_code_predictions_2d(gt_segmentation, pred_segmentation):
    assert len(gt_segmentation.shape) == 2, "Expected 2D gt"
    assert len(pred_segmentation.shape) == 2, "Expected 2D pred"

    masks = {
        "tp": np.logical_and(gt_segmentation, pred_segmentation),
        "tn": np.logical_and(
            np.logical_not(gt_segmentation), np.logical_not(pred_segmentation)
        ),
        "fp": np.logical_and(np.logical_not(gt_segmentation), pred_segmentation),
        "fn": np.logical_and(gt_segmentation, np.logical_not(pred_segmentation)),
    }

    output = np.zeros(gt_segmentation.shape + (3,), dtype=np.uint8)

    for x in ["tp", "fp", "fn", "tn"]:
        color_array = np.array(cm_color_coding_2d[x]).astype(np.uint8)
        output += np.stack([masks[x]] * 3, axis=-1) * color_array

    return output


def color_code_predictions_3d(gt_segmentation, pred_segmentation):
    assert len(gt_segmentation.shape) == 3, "Expected 3D gt"
    assert len(pred_segmentation.shape) == 3, "Expected 3D pred"

    masks = {
        "tp": np.logical_and(gt_segmentation, pred_segmentation),
        "tn": np.logical_and(
            np.logical_not(gt_segmentation), np.logical_not(pred_segmentation)
        ),
        "fp": np.logical_and(np.logical_not(gt_segmentation), pred_segmentation),
        "fn": np.logical_and(gt_segmentation, np.logical_not(pred_segmentation)),
    }

    output = np.zeros(gt_segmentation.shape, dtype=np.uint8)

    for x in ["tp", "fp", "fn", "tn"]:
        output += masks[x].astype(np.uint8) * cm_color_coding_3d[x]

    return output


def color_code_skeleton_intersection(gt_skeleton, pred_segmentation):
    assert len(gt_skeleton.shape) == 3, "Expected 3D gt skeleton"
    assert len(pred_segmentation.shape) == 3, "Expected 3D pred"

    masks = {
        "tp": np.logical_and(gt_skeleton, pred_segmentation),
        "fn": np.logical_and(gt_skeleton, np.logical_not(pred_segmentation)),
    }

    output = np.zeros(gt_skeleton.shape, dtype=np.uint8)

    for x in ["tp", "fn"]:
        output += masks[x].astype(np.uint8) * cm_color_coding_3d[x]

    return output


def color_code_skeleton_dice(
    gt_skeleton, pred_skeleton, gt_segmentation, pred_segmentation
):
    assert len(gt_segmentation.shape) == 3, "Expected 3D gt"
    assert len(pred_segmentation.shape) == 3, "Expected 3D pred"

    masks = {
        "tp": np.logical_and(pred_skeleton, gt_segmentation),
        "fn": np.logical_and(gt_skeleton, np.logical_not(pred_segmentation)),
        "fp": np.logical_and(np.logical_not(gt_segmentation), pred_skeleton),
    }

    output = np.zeros(gt_segmentation.shape, dtype=np.uint8)

    for x in ["tp", "fp", "fn"]:
        output += masks[x].astype(np.uint8) * cm_color_coding_3d[x]

    return output


def colormap_proj_channels(gt_2d, seg_2d):
    assert len(gt_2d.shape) == 3, "Expected 3 channels"
    assert len(seg_2d.shape) == 3, "Expected 3 channels"

    projection_channels = gt_2d.shape[0]

    pred_all = [preprint_img(seg_2d[ch]) for ch in range(projection_channels)]
    gt_all = [preprint_img(gt_2d[ch]) for ch in range(projection_channels)]
    predictions_colormapped = [
        color_code_predictions_2d(gt, pred) for gt, pred in zip(gt_all, pred_all)
    ]
    return predictions_colormapped


def grid_visualize_matrices_plt(
    data_dict, ordered_names, grid_shape, figsize=(30, 30), caption_size=35, **kwargs
):
    fig = plt.figure(figsize=figsize)
    grid = AxesGrid(
        fig,
        111,
        nrows_ncols=grid_shape,
        axes_pad=1,
    )

    plots = [
        grid[idx].imshow(data_dict[key], **kwargs)
        for idx, key in enumerate(ordered_names)
    ]

    for ax in grid:
        ax.set_axis_off()  # Remove back border around image

    for idx, x in enumerate(plots):
        x.axes.get_xaxis().set_visible(False)
        x.axes.get_yaxis().set_visible(False)
        x.axes.set_title(ordered_names[idx], fontsize=caption_size, pad=caption_size)

    return fig


def grid_visualize_matrices_plt_from_list(
    images, captions, grid_shape, figsize=(30, 30), caption_size=35, **kwargs
):
    fig = plt.figure(figsize=figsize)
    grid = AxesGrid(
        fig,
        111,
        nrows_ncols=grid_shape,
        axes_pad=1,
    )

    plots = [grid[idx].imshow(img, **kwargs) for idx, img in enumerate(images)]

    for ax in grid:
        ax.set_axis_off()  # Remove back border around image

    for idx, x in enumerate(plots):
        x.axes.get_xaxis().set_visible(False)
        x.axes.get_yaxis().set_visible(False)
        x.axes.set_title(captions[idx], fontsize=caption_size, pad=caption_size)

    return fig


def display_pixel_matrix(
    cm,
    ticks,
    title=None,
    threshold=None,
    min_val=None,
    max_val=None,
    cmap=None,
    higher_white=False,
):
    figure = plt.figure(figsize=(5, 5), dpi=300)

    # Display colors
    plt.imshow(cm, cmap=cmap, vmin=min_val, vmax=max_val)
    plt.colorbar()

    # Ticks
    y_ticks, x_ticks = ticks
    plt.xticks(np.arange(len(x_ticks)), x_ticks)
    plt.yticks(np.arange(len(y_ticks)), y_ticks)

    # Display values
    if threshold is None:
        bw_display_threshold = cm.max() / 2.0
    else:
        bw_display_threshold = threshold
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if higher_white:
            color = "white" if cm[i, j] > bw_display_threshold else "black"
        else:
            color = "black" if cm[i, j] > bw_display_threshold else "white"
        plt.text(j, i, f"{cm[i, j]:.4f}", horizontalalignment="center", color=color)

    plt.title(title)
    return figure


def plot_samplewise_figure(results_dict, output_file):
    split_order = ["train", "val", "test"]
    metrics = ["skeleton_recall", "dice_3d", "precision_3d", "recall_3d"]

    data = {
        metric: [
            results_dict[f"results_{split}"]["samplewise_metrics"][metric]
            for split in split_order
        ]
        for metric in metrics
    }
    all_scan_ids = flatten_list(
        [results_dict[f"results_{split}"]["scan_ids"] for split in split_order]
    )

    N_train = len(results_dict["results_train"]["scan_ids"])
    N_val = len(results_dict["results_val"]["scan_ids"])
    N_test = len(results_dict["results_test"]["scan_ids"])
    N = len(all_scan_ids)

    x0 = 0
    x1 = N_train
    x2 = N_train + N_val

    split_x = dict(zip(split_order, [x0, x1, x2]))

    color_dataset_sep = "#FFD9DA"
    colors = {
        "skeleton_recall": "#A0CCDA",  # Light blue
        "dice_3d": "#984447",  # Cordovan (red)
        "precision_3d": "#CBD081",  # Citron
        "recall_3d": "#5B6C5D",  # Feldgrau
    }
    plt.figure(figsize=(70, 10))

    plt.title("Samplewise dice | precision | recall | skeleton recall")
    plt.xticks(list(range(0, N)), labels=all_scan_ids, rotation=45)
    plt.yticks(np.arange(0.5, 1.01, 0.05))
    plt.xlim([-1, N])
    plt.ylim([0.5, 1.05])
    for x in range(0, N):
        plt.axvline(x=x, ymax=0.5 / 0.55, color="k", linestyle=":", alpha=0.1)

    for split_name, x_coord in split_x.items():
        plt.text(
            x_coord - 0.25,
            1.04,
            s=split_name,
            rotation=90,
            fontsize=10,
            color=color_dataset_sep,
            verticalalignment="top",
            bbox=dict(facecolor="w", alpha=1, edgecolor=color_dataset_sep),
        )

    for split_name, x_coord in split_x.items():
        plt.axvline(
            x=x_coord - 0.5,
            color=color_dataset_sep,
            linestyle="-",
            alpha=1,
            linewidth=2.5,
        )

    for _, (metric, color) in enumerate(colors.items()):
        values = flatten_list(data[metric])
        plt.plot(values, alpha=0.2, color=color, label=metric)
        plt.scatter(y=values, x=list(range(0, N)), alpha=0.5, color=color)
        for pos, val in enumerate(values):
            plt.text(
                pos + 0.3,
                val,
                s=f"{val:.2f}",
                fontsize=5,
                color="k",
                alpha=0.8,
                bbox=dict(facecolor="w", alpha=1, edgecolor="none"),
            )

        mean_hline_options = dict(color=color, linestyle=":", alpha=0.4, linewidth=1.5)
        plt.axhline(
            y=np.mean(data[metric][0]), xmin=x0, xmax=x1 / N, **mean_hline_options
        )
        plt.axhline(
            y=np.mean(data[metric][1]), xmin=x1 / N, xmax=x2 / N, **mean_hline_options
        )
        plt.axhline(
            y=np.mean(data[metric][2]), xmin=x2 / N, xmax=1, **mean_hline_options
        )

    plt.legend(loc="lower left")
    plt.savefig(output_file, dpi=300)
