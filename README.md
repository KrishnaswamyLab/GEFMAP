# Metabolic Graph Neural Network

## NOTE!!!
We are still cleaning up the code!

## Project Description
The goal of this project is to model metabolic network activity from single cell RNA sequencing (scRNAseq) data. 
Metabolic flux prediction has traditionally been done using a linear optimization method called [flux balance analysis (FBA)](https://pubmed.ncbi.nlm.nih.gov/33292061/). This relies on curated genome-scale metabolic models (GEM) that provide stoichiometry matrices (S matrices) quantifying the ratio of substrates (metabolites moving in) and products (metabolites moving out). A reaction rate vector v represents the rate of flux for each reaction. In order to satisfy mass conservation, the product of this rate vector v and the stoichiometry matrix S must be zero (Sv=0). In other words, the spectrum of all possible metabolic states is defined by the null space of this S matrix. 

For a given system (e.g., individual cell) the user can select a reaction or set of reactions to optimize (maximize flux through). Given this objective function constrained by Sv=0, a linear solver can then predict the reaction rate vector v and flux rate for all reactions in the metabolic network. Other constraints can be added, for example upper and lower bounds for each reaction that depend on gene expression (expression of genes encoding enzymes that catalyze the reaction). 

Gene expression is not a direct indicator of either enzyme activity or reaction flux for a variety of reasons (post-transcriptional regulation of gene expression, post-translational modification of enzymes, metabolite feedback inhibition). Recently, [Compass](https://yoseflab.github.io/software/compass/) was developed for metabolic modeling from scRNAseq and addressed this issue by prediction 'maximal flux $v_{r}^{\text{opt}}$ that every reaction can carry' after imposing upper/lower balance constraints based on gene expression. This functionally identifies reactions and pathways that may be important to cellular differentation and function without needing to predict absolue flux values for the entire system. This, however, requires iteratively running a linear optimizer and can take weeks to run on a standard scRNAseq dataset. 

We aim to improve both the computational efficiency and biological relevance of these methods using a deep learning approach. Because metabolic networks naturally exist as [fully connected, directed, weighted graphs](https://www.nature.com/articles/s41540-018-0067-y#:~:text=Mass%20flow%20graphs%3A%20incorporating%20information,predict%20environment%2Dspecific%20flux%20distributions) we have developed a supervised graph neural network to predict flux balance solutions given an input objective function and gene expression (represented as reaction upper and lower bounds). The network is trained on previously computed FBA solutions, and the loss function includes a penalty to enforce mass conservation and stoichiometric balance (Sv=0).

Although steady-state levels of gene expression poorly indicate reaction rates, we can assume with relatively high confidence that if a cell is actively upregulating a set of genes involved in a metabolic process during differentiation or changes in function, it is actively trying to increase activity of that metabolic process. Therefore, we aim to model transcriptional dynamics across time (physical time or inferred developmental time) in order to define an objective function. Here, we use a [hybrid graph scattering network](https://arxiv.org/abs/2206.01506) designed to address the max clique problem that has been adapted to predict a max <i>weighted</i> clique, where weights represent increases in gene expression from a prior cell state. We then consider the reactions within this max weighted clique to comprise the objective function, which we can optimize for using either FBA or our GNN. 

<b>Ongoing work: </b>Although our current GNN works on undirected weighted graphs, we are now working to adapt it using a recently defined version of the Laplace-Bellatrami Operator for directed graphs called the [Magnetic Laplacian](https://github.com/matthew-hirn/magnet). 


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
