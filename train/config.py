from dataclasses import dataclass, fields
from typing import List


def serialize_class_attributes(cls, indent="  ", attr_spacer="\n", attr_separator=","):
    """Prints the attributes of a class."""
    field_names = [field.name for field in fields(cls)]
    field_values = [getattr(cls, field) for field in field_names]
    fields_string = f'{attr_separator.join([f"{attr_spacer}{indent*2}{x}: {y}" for x, y in zip(field_names, field_values)])}'
    serialized_str = (
        f"{cls.__class__.__name__}({fields_string}{attr_separator}{attr_spacer})"
    )
    return serialized_str


@dataclass
class GlobalConfig:
    debug: bool
    inference: bool

    timestamp: str
    experiment_folder: str
    subfolder: str
    experiment_name: str

    wandb_group: str
    wandb_tags: List[str]
    wandb_notes: str

    epochs: int
    iterations: int
    iteration_based: bool  # If true, will ignore epochs and use iterations instead
    validation_frequency: int

    seed: int
    lr: float

    alpha_depth: float
    alpha_2d: float
    alpha_3d: float

    def __repr__(self):
        return serialize_class_attributes(self)

    def __str__(self):
        return serialize_class_attributes(self)


@dataclass
class ModelConfig:
    # image_size: List[int]
    first_channels: int
    levels: int
    dropout_encoder: bool
    dropout_decoder: bool
    dropout_rate: int
    dropout_depth: int
    concatenation: bool
    load_model_source: str
    load_model_metric: str

    def __repr__(self):
        return serialize_class_attributes(self)

    def __str__(self):
        return serialize_class_attributes(self)


@dataclass
class DatasetConfig:
    split_file: str
    batch_size: int
    num_workers: int
    projection_viewpoints: List[List[int]]
    viewpoint_assignment: str
    depth_file: str

    def __repr__(self):
        return serialize_class_attributes(self)

    def __str__(self):
        return serialize_class_attributes(self)


@dataclass
class Config:
    global_conf: GlobalConfig
    model_conf: ModelConfig
    dataset_conf: DatasetConfig

    def __init__(self, config):
        self.global_conf = GlobalConfig(**config["global"])
        self.model_conf = ModelConfig(**config["model"])
        self.dataset_conf = DatasetConfig(**config["dataset"])

        # Input validation for empty variables
        if self.dataset_conf.projection_viewpoints == "None":
            self.dataset_conf.projection_viewpoints = None
