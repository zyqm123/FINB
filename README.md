# FINB: Core Code and Datasets

This repository provides the official implementation of the FINB algorithm, as introduced in our paper. The code includes the essential components for dataset preparation, training, and evaluation.

## Installation
Please install the required dependencies by running:

```
pip install -r requirements.txt
```
We recommend using Python ≥3.8 and CUDA ≥11.0 for GPU acceleration.

## Datasets
The experiments follow the standard Cross-Domain Few-Shot Learning (CD-FSL) setting:

* Source domain: mini-ImageNet

* Target domains: CUB, Cars, Places, Plantae

Dataset setup and preprocessing are consistent with the [FWT](https://github.com/hytseng0509/CrossDomainFewShot) repository.

## Code Structure

The core implementation is organized as follows:

* FINB_main.py：Entry point for training and evaluation. Loads datasets, initializes pretrained models, and performs meta-training.

* FINB.py：Implements the FINB algorithm. Includes intermediate domain construction and cross-domain knowledge transfer.

## Usage

To evaluate FINB on a target domain, run:

```
python FINB_main.py --config FINB.yaml
```

The configuration file specifies experimental settings, including datasets, model parameters, and training protocols.

## Reproducing Results

This repository contains the core implementation used in our paper.
The complete codebase, including additional modules and utilities, will be released upon acceptance of the paper.
