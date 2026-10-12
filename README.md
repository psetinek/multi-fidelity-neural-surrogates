# Compute-optimal data scaling for neural surrogates via multi-fidelity training

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-2d6a4f.svg)](https://psetinek.github.io/multi-fidelity-neural-surrogates/)
[![Paper](https://img.shields.io/badge/arXiv-2506.12007-b31b1b.svg)](https://neurips.cc/virtual/2026/loc/sydney/poster/151246)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-multi--fidelity--airfoil-ffd21e.svg)](https://huggingface.co/datasets/psetinek/multi-fidelity-airfoil)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

![Figure 1](./assets/fig_1.png)

</div>

> **Paul Setinek\*, Fabian Paischer, Nils Thuerey, Johannes Brandstetter, Felix Koehler\***

This is the official repository for the paper "Compute-optimal data scaling for neural surrogates via multi-fidelity training", accepted at [NeurIPS 2026](https://neurips.cc/virtual/2026/loc/sydney/poster/151246).

## Repository structure

We split the grid and mesh experiments into two different folders containing all experiments of the paper:

| folder | experiments | architectures |
|---|---|---|
| [`grid/`](grid) | Poisson equation on 1D, 2D and 3D grids (discretization and solver error) | FNO, U-Net |
| [`mesh/`](mesh) | linear elasticity and hyperelasticity on 2D meshes (discretization and solver error), RANS airfoils (discretization error) | Transolver, AB-UPT |

Each folder is self-contained, with its own installation instructions, data generation, training
and evaluation code, and the results and scripts behind its figures in the paper.

## Data

The airfoil dataset (1,627 cases, 11 mesh levels each, ~107 GB) is available on [Hugging Face](https://huggingface.co/datasets/psetinek/multi-fidelity-airfoil).
The other datasets are generated with the pipelines in the respective folder and can be regenerated on the fly or stored locally.

## Citation

```bibtex
@inproceedings{setinek2026multifidelity,
  title     = {Compute-optimal data scaling for neural surrogates via multi-fidelity training},
  author    = {Setinek, Paul and Paischer, Fabian and Thuerey, Nils and Brandstetter, Johannes and Koehler, Felix},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
