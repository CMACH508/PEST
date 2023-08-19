# PEST: A General-Purpose Protein Embedding Model for Homology Search

## Introduction

In this paper, we propose a novel general-purpose protein embedding model that can be used for homology search. It first employs a protein language pre-training model to extract protein sequence embedding, capturing intricate biological patterns. Subsequently, a Transformer with an attention module integrating protein structural information generates the high-level protein representations. By combining protein sequence and structural features, the model can effectively exploit the rich contextual and spatial information inherent in proteins. We applied the model to the SCOP dataset for protein superfamily classification, achieving a classification accuracy of 86.97%, outperforming state-of-the-art methods by 7.91%. 

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

python train.py

python test.py

