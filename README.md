# GEFMAP


## Introduction

**GEFMAP** (**G**ene **E**xpression-based **F**lux **M**apping and **M**etabolic pathway **P**rediction) combines a dual system of complementary neural networks that uses bulk or single-cell gene expression data to both (1.) infer a cellular metabolic objective and (2.) estimate stead-state reaction flux rates across biochemical networks. For a full description of our methods and applications, please see our [preprint at bioRxiv](https://t.co/7WJT46ka5W), accepted for publication at [RECOMB 2024](https://recomb.org/recomb2024/index.html).


## Overview

![graphical abstract methods overview](/images/github/overview.png)

**i. Metabolic Graph Construction** Given a network with $m$ metabolites and $n$ reactions, we construct an (undirected) graph $G=(V,E)$ where nodes $V$ represent reactions. We define edges $\{v_i, v_j\}\in E$ if reactions share a common metabolite in a consumer/producer relationship and edge weights represent metabolite flux, or the amount of metabolite moving between nodes $v_i$ and $v_j$ per arbitrary unit of time. Node weights are then defined as $\Delta$ gene expression (mapped from genes to reactions) relative to either a control sample or initial timepoint. By using dynamic rather than static gene expression, we interpret these node weights to be a measure of how much a cell (or transcriptional program) is attempting to actively engage a metabolic reaction.

**ii. Cellular Metabolic Objective Function GNN** The first subnetwork  <i>infers the cellular metabolic objective</i>  based on the intuition that the cell upregulates expression of catalytic enzymes (genes) for producing its desired metabolic state. Here, we formulate this as the problem of finding a highly-weighted, highly-connected subgraph in the metabolic network graph where the nodes representing individual reactions are given weights according to the expression levels of associated genes. This allows us to essentially infer the cellular objective from its transcriptomic profile. To do this, we utilize a deep neural network based on the geometric scattering transform to estimate a large highly-connected subnetwork by solving a maximum weighted subgraph, a relaxed version of the maximum weighted clique problem. We then formulate a cellular objective function corresponding to maximizing the reactivity in this subgraph.

**iii. Reaction Flux Estimation (Null Space) Nerual Network** Our second subnetwork <i>solves the cellular  objective</i> by identifying a set of reaction rates $\mathbf{v}$ that maximizes the objective, given the constraint that our solution $\mathbf{v}$ must satisfy mass balance within the system. We impose this using a matrix of reaction stoichiometries $S$, where a solution $\mathbf{v}$ satisfies the mass balance constraint if $S\mathbf{v}=0$, inspired by flux balance analysis (FBA). We therefore consider a basis for the null space, $S\mathbf{v}=0$, and design a novel network that operates in this null space to find the coefficients of the solution with respect to this basis. Thus, GEFMAP is able to utilize both the structure of the network and the geometric constraint that $\mathbf{v}$ lies within the null space of $S$ to predict the metabolic fluxes. In essence this allows us to predict the entire metabolic state based on the inferred objective, which will include maximizing reactions of a subnetwork and may have pervasive effects on system-wide flux. 

## Dependencies
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
conda create --name metabolic_graph pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda activate metabolic_graph
conda install pytorch_geometric torch-scatter pytorch-lightning -c conda-forge
python -m pip install pysmiles graphein phate
conda install pytorch3d -c pytorch3d
conda install scikit-image pillow -c anaconda
python -m pip install git+https://github.com/KrishnaswamyLab/Multiscale_PHATE
```

## Usage
```
conda activate metabolic_graph
cd src/
## TO BE ADDED
```
