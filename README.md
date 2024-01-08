# 3Dseg-mip-depth
Code for "3D Arterial Segmentation via Single 2D Projections and Depth Supervision in Contrast-Enhanced CT Images" @MICCAI 2023

### 3D Arterial Segmentation via Single 2D Projections and Depth Supervision in Contrast-Enhanced CT Images

Paper: https://arxiv.org/pdf/2309.08481.pdf

Supplementary material: https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-031-43907-0_14/MediaObjects/549755_1_En_14_MOESM1_ESM.pdf

<img title="Overview" alt="Overview" src="graphical_abstract.jpg">


## Installation instructions

1. Set up the new environment

    a. Create new environment and install main packages
    ```
      mamba env create -f environment.yml
    ```
    b. Load environment
    ```
      conda activate 3Dseg-mip-depth
    ```
    c. Install additional packages
      ```
      pip install torch-sparse torch-scatter torch-geometric
      pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1131/download.html
      git clone https://github.com/deepmind/surface-distance.git
      pip install surface-distance/
      ```

## Run instructions

Will follow





## Please cite this work as: 
```
@inproceedings{dima20233d,
  author = {Alina F. Dima and Veronika A. Zimmer and Martin J. Menten and Hongwei Bran Li and Markus Graf and Tristan Lemke and Philipp Raffler and Robert Graf and Jan S. Kirschke and Rickmer Braren and Daniel Rueckert},
  title = {3D Arterial Segmentation via Single 2D Projections and Depth Supervision in Contrast-Enhanced {CT} Images},
  booktitle = {Medical Image Computing and Computer Assisted Intervention - {MICCAI}},
  pages = {141--151},
  publisher = {Springer},
  year = {2023},
  doi = {10.1007/978-3-031-43907-0\_14},
}

```