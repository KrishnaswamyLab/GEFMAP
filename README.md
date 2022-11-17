# Metabolic Graph Neural Network

## NOTE!!!
We are still cleaning up the code!

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
