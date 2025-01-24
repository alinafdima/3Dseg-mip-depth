import logging
import math
import os
from glob import glob
from os.path import join
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

import visualization as vis
from affine_utils import get_inverse_affine_matrix_tio_convention, resample_image
from utils import set_random_seed

from config import Config, ModelConfig
from dataset import DepthDataset, InferenceDataset
from evaluator import EvaluatorTraining, InferenceEvaluator
from projector_2D import project_batch
from unet import UNet3D

log = logging.getLogger(__name__)

image_shape = (256, 256, 128)


# #############################################################################################
# Netowork architecture
# #############################################################################################


def init_model(cfg: ModelConfig):
    log.info("Initializing model...")

    model = UNet3D(
        inchannels=1,
        outchannels=1,
        first_channels=cfg.first_channels,
        image_size=image_shape,
        levels=cfg.levels,
        dropout_enc=cfg.dropout_encoder,
        dropout_dec=cfg.dropout_decoder,
        dropout_rate=cfg.dropout_rate,
        dropout_depth=cfg.dropout_depth,
        concatenation=cfg.concatenation,
    )

    return model.cuda()


class Trainer:

    # #############################################################################################
    # Init and setup
    # #############################################################################################

    def __init__(self, cfg: DictConfig, wandb_name: str, experiment_folder: str):
        self.cfg = cfg

        # General configuration settings
        gconf = cfg.global_conf
        self.gconf = gconf
        self.wandb_name = wandb_name

        self.dimensionalities = ["2d", "3d"]
        self.metrics_categories_train = ["dice"]
        self.metrics_categories_inference = self.metrics_categories_train + [
            "confusion_matrix_based",
            "connected_components",
            "surface_based",
        ]
        self.alpha_depth = gconf.alpha_depth
        self.alpha_2d = gconf.alpha_2d
        self.alpha_3d = gconf.alpha_3d

        self.set_up_paths_and_folders(experiment_folder)

        # Init data loaders
        self.init_dataloaders()

        # Init model
        set_random_seed(gconf.seed)
        self.model = init_model(cfg.model_conf)

        # Loss, optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), gconf.lr)

        # Load weights
        self.load_model()

    def set_up_paths_and_folders(self, experiment_folder):
        self.experiment_folder = experiment_folder
        self.models_folder = join(self.experiment_folder, "checkpoints")
        self.metrics_folder = join(self.experiment_folder, "metrics")

        for folder in [self.models_folder, self.metrics_folder]:
            Path(folder).mkdir(parents=False, exist_ok=True)

        # Set up metrics
        self.metrics_files = {
            mode: {
                suffix: join(self.metrics_folder, f"{mode}_metrics_{suffix}.csv")
                for suffix in ["averages", "sample_wise"]
            }
            for mode in ["train", "val", "test"]
        }
        for mode in ["train", "val", "test"]:
            self.metrics_files[mode]["inference"] = join(
                os.environ["EXPERIMENTS_ROOT"],
                "inference_all",
                f"{mode}_metrics_inference.csv",
            )

    # #############################################################################################
    # Data loaders
    # #############################################################################################

    def init_dataloaders(self):
        cfg = self.cfg.dataset_conf
        param_dict = dict(
            cv_file=cfg.split_file,
            depth_file=cfg.depth_file,
            viewpoint_assignment=cfg.viewpoint_assignment,
            projection_viewpoints=cfg.projection_viewpoints,
        )

        ds = {
            "train": DepthDataset(**param_dict, split="train", data_augmentation=True),
            "val": DepthDataset(**param_dict, split="val", data_augmentation=False),
            "train_inf": InferenceDataset(cv_file=cfg.split_file, split="train"),
            "val_inf": InferenceDataset(cv_file=cfg.split_file, split="val"),
            "test": InferenceDataset(cv_file=cfg.split_file, split="test"),
        }
        get_dataloader = lambda ds, batch_size, shuffle: DataLoader(
            ds,
            batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            collate_fn=ds.collate_function,
        )
        self.dataset = ds
        self.dl = {
            "train": get_dataloader(ds["train"], cfg.batch_size, shuffle=True),
            "train_inf": get_dataloader(ds["train_inf"], 1, shuffle=False),
            "val": get_dataloader(ds["val"], 1, shuffle=False),
            "val_inf": get_dataloader(ds["val_inf"], 1, shuffle=False),
            "test": get_dataloader(ds["test"], 1, shuffle=False),
            "test_inf": get_dataloader(ds["test"], 1, shuffle=False),
        }

    # #############################################################################################
    # Train / Eval / Test logic
    # #############################################################################################

    def configure_training_epoch(self):
        """
        Configures the total number of training epochs.
        If the training is iteration-based, the number of training epochs is set to the total number of iterations divided by the number of batches in the training set.
        Otherwise it will set the number of training epochs according to the config file.
        It will also log some info regarding validation frequency and total number of epochs.

        Example:
            gconf.iterations = 105
            gconf.iteration_based = True
            batches = 20
          The model will train for ceil(105/20) = 6 epochs.
        """
        if self.gconf.iteration_based:
            self.total_epochs = math.ceil(self.gconf.iterations / len(self.dl["train"]))
        else:
            self.total_epochs = self.gconf.epochs
        self.last_epoch = self.global_epoch + self.total_epochs

        log.info(
            f"Trainig starting at epoch {self.global_epoch + 1} until {self.last_epoch}, for a total of {self.total_epochs} epochs."
        )

        if self.gconf.iteration_based:
            log.info(
                f'Validating approximately every {self.gconf.validation_frequency / len(self.dl["train"])} epochs.'
            )
        else:
            log.info(
                f"Validating approximately every {self.gconf.validation_frequency} epochs."
            )

    def is_validation_epoch(self) -> bool:
        """
        Checks whether it should perform validation this epoch.
        It validates periodically either based on the amount of iterations or epochs, but always at the end of an epoch.
        And it also validates at the end of the training.

        Example:
          gconf.iterations = 105
          gconf.iteration_based = True
          gconf.validation_frequency = 50
          batches = 20

        The model will validate after epochs 3 (60 iterations), 5 (100 iterations) and at the last epoch 6 (120 iterations).

        Returns:
            bool: Whether it is a validation epoch
        """
        if self.last_epoch == self.global_epoch:
            return True

        if self.gconf.iteration_based:
            if self.perform_validation:
                self.perform_validation = False
                return True
        else:
            return self.global_epoch % self.gconf.validation_frequency == 0

    def train_model(self):
        self.configure_training_epoch()
        self.evaluator = {}
        self.perform_validation = False

        for mode in ["train", "val"]:
            track_scan_id = self.dataset[mode].scan_ids[0]
            self.evaluator[mode] = EvaluatorTraining(
                metrics_categories=self.metrics_categories_train,
                mode=mode,
                metrics_files=self.metrics_files[mode],
                track_ids=[track_scan_id],
                output_folder=join(self.experiment_folder, f"track_{mode}"),
            )

        self.evaluator["val"].init_global()

        for _ in range(self.total_epochs):
            self.global_epoch += 1

            ########################################
            # Training
            log.info(f"Training epoch {self.global_epoch}...")
            self.model.train()
            self.evaluator["train"].init_epoch()

            for _, sample in tqdm(enumerate(self.dl["train"])):
                self.global_step += 1
                if self.global_step % self.gconf.validation_frequency == 0:
                    self.perform_validation = True
                tensors_npy, loss_dict = self.network_pass(sample, backprop=True)
                self.evaluator["train"].log_batch(
                    tensors_npy, sample, loss_dict, self.global_epoch
                )

            self.evaluator["train"].finalize_epoch(self.global_epoch)

            ########################################
            # Validation
            if self.is_validation_epoch():
                log.info(f"Validation epoch {self.global_epoch}...")
                self.model.eval()
                self.evaluator["val"].init_epoch()

                for _, sample in tqdm(enumerate(self.dl["val"])):
                    tensors_npy, loss_dict = self.network_pass(sample, backprop=False)
                    self.evaluator["val"].log_batch(
                        tensors_npy, sample, loss_dict, self.global_epoch
                    )

                epoch_dict = self.evaluator["val"].finalize_epoch(self.global_epoch)

                # Update saved models
                dict_updated = self.evaluator["val"].update_global(
                    epoch_dict, self.global_epoch
                )
                for key, best_value in dict_updated.items():
                    self.save_model(key, best_value)

        ########################################
        # Inference at the end of training
        self.load_model(use_current_run=True)
        self.global_epoch = (
            self.last_epoch + 1
        )  # Since I can't update a past epoch with wandb
        self.run_inference(data_splits=["test", "val", "train"])

    def run_inference(self, data_splits: List[str]):
        print("Running inference...")
        self.model.eval()
        results = {}
        for data_split in data_splits:
            best_epoch = self.best_epoch
            evaluator = InferenceEvaluator(
                best_epoch, data_split, self.experiment_folder, run=None
            )
            dataloader = self.dl[f"{data_split}_inf"]
            for _, sample_dict in tqdm(enumerate(dataloader)):
                self.optimizer.zero_grad()
                tensors = {key: sample_dict[key].cuda() for key in ["input", "gt_3d"]}
                tensors["pred_3d"] = self.model(tensors["input"].cuda())
                tensors_npy = {
                    key: tensors[key].cpu().detach().numpy()
                    for key in ["input", "gt_3d", "pred_3d"]
                }
                tensors_npy["seg_3d"] = (tensors_npy["pred_3d"] > 0.5).astype(int)
                evaluator.log_batch(tensors_npy, sample_dict)
            results[f"results_{data_split}"] = evaluator.finalize(self.global_epoch)

        vis.plot_samplewise_figure(
            results,
            output_file=join(self.experiment_folder, "summary_samplewise_dice.png"),
        )

    # #############################################################################################
    # Forward / backward pass
    # #############################################################################################

    def network_pass(self, sample, backprop):
        self.optimizer.zero_grad()
        tensors = {key: sample[key].cuda() for key in ["input", "gt_3d", "depth"]}

        network_output = self.model(tensors["input"].cuda())

        out_tensors = []
        batch_size = network_output.shape[0]
        for im_id in range(batch_size):
            out = network_output[im_id, ...].unsqueeze(0)
            affine_params = sample["affine_params"][im_id]
            if affine_params is not None:
                A_inverse = get_inverse_affine_matrix_tio_convention(
                    image_shape, affine_params
                )
                out_nonaffine = resample_image(
                    out,
                    A_inverse,
                    interpolation="bilinear",
                    requires_grad=True,
                    cuda=True,
                )
                out_tensors.append(out_nonaffine)
            else:
                out_tensors.append(out)

        tensors["pred_3d_affine"] = network_output
        tensors["pred_3d"] = torch.concat(out_tensors, dim=0)
        tensors["gt_2d"] = sample["gt_projections"].cuda()
        tensors["pred_2d"], tensors["pred_vp_rotated"] = project_batch(
            tensors["pred_3d"],
            sample["A_projections"],
            image_shape,
            interpolation="bilinear",
        )
        tensors["pred_2d"] = tensors["pred_2d"].cuda()
        tensors["pred_vp_rotated"] = tensors["pred_vp_rotated"].cuda()

        # import ipdb; ipdb.set_trace()
        loss_depth = self.criterion(
            tensors["pred_vp_rotated"] * tensors["depth"], tensors["depth"]
        )
        loss_2d = self.criterion(tensors["pred_2d"], tensors["gt_2d"])
        loss_3d = self.criterion(tensors["pred_3d"], tensors["gt_3d"])

        loss = (
            self.alpha_depth * loss_depth
            + self.alpha_2d * loss_2d
            + self.alpha_3d * loss_3d
        )

        with torch.no_grad():
            loss_depth = loss_depth.cpu().item()
            loss_2d = loss_2d.cpu().item()
            loss_3d = loss_3d.cpu().item()
            loss_value = loss.cpu().item()
            loss_dict = {
                "loss": loss_value,
                "loss_depth": loss_depth,
                "loss_2d": loss_2d,
                "loss_3d": loss_3d,
            }

        # Backpropagation
        if backprop:
            loss.backward()
            self.optimizer.step()

        # Detach tensors for computing metrics and saving outputs
        tensors_npy = {
            key: tensors[key].cpu().detach().numpy()
            for key in [
                "input",
                "gt_3d",
                "pred_3d",
                "pred_3d_affine",
                "gt_2d",
                "pred_2d",
            ]
        }
        tensors_npy["seg_2d"] = (tensors_npy["pred_2d"] > 0.5).astype(int)
        tensors_npy["seg_3d"] = (tensors_npy["pred_3d"] > 0.5).astype(int)
        tensors_npy["seg_3d_affine"] = (tensors_npy["pred_3d_affine"] > 0.5).astype(int)

        return tensors_npy, loss_dict

    # #############################################################################################
    # Model saving and loading
    # #############################################################################################

    def save_model(self, key, value):
        epoch = self.global_epoch
        log.info(
            f" --> Updated {key} model at epoch {epoch}. New value: {value[0]:.4f}"
        )

        if key == "dice_2d_best":
            self.best_epoch = epoch

        model_path = join(self.models_folder, f"{key}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                key: value,
            },
            model_path,
        )

    def load_model(self, use_current_run=False):
        cfg = self.cfg.model_conf

        if use_current_run:
            metric = cfg.load_model_metric
            load_model_source = join(self.models_folder, f"{metric}_best.pth")
            self.load_model_from_file(load_model_source)

        else:
            if cfg.load_model_source == "":
                log.info("No checkpoint used, training from scratch...")
                self.global_step = 0
                self.global_epoch = 0
                self.best_epoch = 0
            else:
                log.info("Loading model from source...")

                source_name = cfg.load_model_source
                metric = cfg.load_model_metric

                # Find source model folder
                matches = glob(
                    join(os.environ["EXPERIMENTS_ROOT"], "*", "**", f"*{source_name}*"),
                    recursive=True,
                )
                if not len(matches) == 1:
                    log.info(f"Found the following experiment matches {matches} ..:(")
                    raise ValueError(
                        f"Found {len(matches)} experiment matches for {cfg.load_model_source}!"
                    )
                load_model_source = join(
                    matches[0], "checkpoints", f"{metric}_best.pth"
                )

                self.load_model_from_file(load_model_source)

    def load_model_from_file(self, load_model_source):
        # Load checkpoint
        saved_state = torch.load(load_model_source)
        self.model.load_state_dict(saved_state["model_state_dict"])
        self.optimizer.load_state_dict(saved_state["optimizer_state_dict"])
        epoch = saved_state["epoch"]

        # Initialize global step and epoch
        self.global_step = epoch * len(self.dl["train"])
        self.global_epoch = epoch
        self.best_epoch = epoch

        log.info(f"Loaded model from {load_model_source} at epoch {epoch}.")

    # #############################################################################################


@hydra.main(version_base=None, config_path=".", config_name="cfg")
def main(cfg_raw: DictConfig) -> None:

    cfg = Config(cfg_raw)
    is_inference = cfg.global_conf.inference
    if is_inference:
        log.info("Running inference with config:")
    else:
        log.info("Training with config:")
    log.info(cfg)

    # Please configure experiment folders
    experiment_folder = "test"
    wandb_name = "random-monkey-123"

    if is_inference:
        Trainer(cfg, wandb_name, experiment_folder).run_inference(["test", "val"])
    else:
        Trainer(cfg, wandb_name, experiment_folder).train_model()

    log.info("Finished")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
