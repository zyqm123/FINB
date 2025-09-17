This repository contains the official implementation and datasets for the FINB algorithm, as presented in our paper.

Installation
Install the required dependencies using:

bash
pip install -r requirements.txt
Datasets
We use mini-ImageNet as the single source dataset, and four target datasets: CUB, Cars, Places, and Plantae.
For dataset setup and preprocessing, please refer to the CrossDomainFewShot repository.

Code Overview
The core implementation is organized into two main components:

FINB_pretraining.py: Code for model training

FINB_finetuning: Code for meta-testing and evaluation

Usage
Training
To train the FINB model, run:

bash
python FINB_pretraining.py --config configs/FINB.yaml
Experiment parameters can be modified in configs/FINB.yaml.

Testing
To evaluate the model, run:

bash
python FINB_finetuning.py --config configs/meta_test.yaml
Reproducing Results
To reproduce the results reported in the paper, execute the above training and testing commands in sequence.