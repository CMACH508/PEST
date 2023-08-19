# PEST: A General-Purpose Protein Embedding Model for Homology Search

This repository contains the source code, trained models and the some test examples for PEST.

## Introduction

In this paper, we propose a novel general-purpose protein embedding model that can be used for homology search. It first employs a protein language pre-training model to extract protein sequence embedding, capturing intricate biological patterns. Subsequently, a Transformer with an attention module integrating protein structural information generates the high-level protein representations. By combining protein sequence and structural features, the model can effectively exploit the rich contextual and spatial information inherent in proteins. We applied the model to the SCOP dataset for protein superfamily classification, achieving a classification accuracy of 86.97%, outperforming state-of-the-art methods by 7.91%. 

## Overview

<img src=".\fig\overview.png" width="100%" />

## Requirements

```
pip install -r requirements.txt
```

## Dataset

Step 1: Download the SCOP dataset (https://scop.berkeley.edu/astral/pdbstyle/ver=2.08) to the 'dataset' directory.

Step 2: Process the 'ent' files into JSON format using DSSP (preprocess.py).

Step 3: Divide the dataset into training, validation, and test sets.

Step 4: Prepare the data (prepare.py).

## Train and test

To train PEST, run the command as follows:

```
python train.py
```

To test PEST, run the command as follows:

```
python test.py
```

## Trained model

Download the trained model from Zenodo(https://doi.org/10.5281/zenodo.8265821) to the 'checkpoints' directory.