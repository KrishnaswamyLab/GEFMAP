# Metabolic Graph Neural Network

## NOTE!!!
We are still cleaning up the code!

Notebooks for metabolic models and data preprocessing can be found at the shared [google drive folder](https://drive.google.com/drive/folders/1xYN6CifcINQaI3TsZzKrN-fHZnKBHuyD?usp=sharing)

## Project Description
The goal of this project is to model metabolic network activity from single cell RNA sequencing (scRNAseq) data. 
Metabolic flux prediction has traditionally been done using a linear optimization method called [flux balance analysis (FBA)](https://pubmed.ncbi.nlm.nih.gov/33292061/). This relies on curated genome-scale metabolic models (GEM) that provide stoichiometry matrices (S matrices) quantifying the ratio of substrates (metabolites moving in) and products (metabolites moving out). A reaction rate vector $v$ represents the rate of flux for each reaction. In order to satisfy mass conservation, the product of this rate vector $v$ and the stoichiometry matrix $S$ must be zero: $Sv=0$. In other words, the spectrum of all possible metabolic states is defined by the null space of this $S$ matrix. 

For a given system (e.g., individual cell) the user can select a reaction or set of reactions to optimize (maximize flux through). Given this objective function constrained by $Sv=0$, a linear solver can then predict the reaction rate vector v and flux rate for all reactions in the metabolic network. Other constraints can be added, for example upper and lower bounds for each reaction that depend on gene expression (expression of genes encoding enzymes that catalyze the reaction). 

Gene expression is not a direct indicator of either enzyme activity or reaction flux for a variety of reasons (post-transcriptional regulation of gene expression, post-translational modification of enzymes, metabolite feedback inhibition). Recently, [Compass](https://yoseflab.github.io/software/compass/) was developed for metabolic modeling from scRNAseq and addressed this issue by prediction 'maximal flux $v_{r}^{\text{opt}}$ that every reaction can carry' after imposing upper/lower balance constraints based on gene expression. This functionally identifies reactions and pathways that may be important to cellular differentation and function without needing to predict absolue flux values for the entire system. This, however, requires iteratively running a linear optimizer and can take weeks to run on a standard scRNAseq dataset. 

We aim to improve both the computational efficiency and biological relevance of these methods using a deep learning approach. Because metabolic networks naturally exist as [fully connected, directed, weighted graphs](https://www.nature.com/articles/s41540-018-0067-y#:~:text=Mass%20flow%20graphs%3A%20incorporating%20information,predict%20environment%2Dspecific%20flux%20distributions) we have developed a supervised graph neural network to predict flux balance solutions given an input objective function and gene expression (represented as reaction upper and lower bounds). The network is trained on previously computed FBA solutions, and the loss function includes a penalty to enforce mass conservation and stoichiometric balance-- $Sv=0$.

**max clique (unweighted)**

i. **Embedding** <br>
$\text{(mlp)} m_{\text{emb}}: X \in \mathbb{R}^{n \times d} \rightarrow H_{0} \in \mathbb{r}^{n \times d_h}$ <br>
$d=3, d_h = 8$

ii. **Diffusion** <br>
$F =  \{f_{lp}, f_{bp}\}$ 
- $f_{lp, r}(H^{\ell-1}) = A^{r}H^{\ell-1} $ <br>
where $A := (D+I)^{-\frac{1}{2}}(W+I)(D+I)^{-\frac{1}{2}}$
- $f_{bp, j}(H^{\ell-1}) = \Psi_{k}H^{\ell-1} $ <br>
where $P := \frac{1}{2}(I_{n}+WD^{-1}) $ <br>
$\Psi_{0} := I_{n} -P $  <br>
$\Psi_{k} := P^{2^{k-1}} - P^{2^{k}}$


$H_{0} \in \mathbb{r}^{n \times d_h}$ <br>
(for $K \in \mathbb{N}$ iterations:)<br>

attention scores: $s^{\ell}_{f} := \sigma(H^{\ell}_{f} || H^{\ell-1})a^{\ell}$ where $s^{\ell}_{f} \in \mathbb{R}^{n}$ <br>
$a_{f}(\upsilon)= \text{SOFTMAX}_{F}s_{f}(\upsilon) \rightarrow$ store in attention vector $\alpha^{\ell}_{f} \in \mathbb{R}^{n}$

*apply $\sigma (.)$ non-linearity before attention vector $a \in \mathbb{R}^{2d_h}$
<br>s.t.<br>
$H_{\text{agg}} := \sum_{f \in F} \alpha^{\ell} \odot H^{\ell}_{f}$ where $\odot$ represents element-wise multiplication
<br>
transform <br>
$\text{(mlp)} m^{\ell}: \mathbb{R}^{d_h} \rightarrow \mathbb{R}^{d_h}$ (leaky RelU)

store results $R = \{H^{\ell} \}^{K}_{\ell =0}$
<br>

iii. **Output** <br>

$H_{\text{cat}} := ||^{k}_{\ell =0} H^{\ell} \in \mathbb{R}^{n \times d_h}$ <br>
$\text{(mlp)} m_{\text{out}}: \mathbb{R}^{d_h \times (K+1)} \rightarrow \mathbb{R}$<br>
$\text{min/max}(h_{\text{out}}) := p$


iv. **LOSS** <br>
for $G = (V,E)$ let $W =$ boolean Adj matrix <br>
- $L_{1}(p) := -p^{T}Wp$ <br>
max that p is on highly connected nodes
<br>

complement graph $\overline{G} = (V, \overline{E})$ 
- $L_{2}(p) := p^{T}\overline{W}p$ <br>
max that p contained within a clique
<br>

such that
- $L(p):= L_1(p) + \beta L_2(p) = -p^{T}Wp + \beta p^{T}\overline{W}p$

Although steady-state levels of gene expression poorly indicate reaction rates, we can assume with relatively high confidence that if a cell is actively upregulating a set of genes involved in a metabolic process during differentiation or changes in function, it is actively trying to increase activity of that metabolic process. Therefore, we aim to model transcriptional dynamics across time (physical time or inferred developmental time) in order to define an objective function. Here, we use a [hybrid graph scattering network](https://arxiv.org/abs/2206.01506) designed to address the max clique problem that has been adapted to predict a max <i>weighted</i> clique, where weights represent increases in gene expression from a prior cell state. We then consider the reactions within this max weighted clique to comprise the objective function, which we can optimize for using either FBA or our GNN. 

<b>Ongoing work: </b>Although our current GNN works on undirected weighted graphs, we are now working to extend it to directred graphs by adapting methods from [MagNet](https://github.com/matthew-hirn/magnet), a GNN that performs spectral convolution on directed graphs using a complex Hermition matrix termed the magnetic Laplacian. 


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
