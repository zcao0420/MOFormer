# MOFormer

<strong>MOFormer: Self-Supervised Transformer model for Metal-Organic Framework Property Prediction</strong> </br>
<em>Journal of the American Chemical Society (JACS)</em> [[Paper]](https://pubs.acs.org/doi/10.1021/jacs.2c11420) [[arXiv]](https://arxiv.org/abs/2210.14188) [[PDF]](https://arxiv.org/pdf/2210.14188.pdf) </br>
[Zhonglin Cao*](https://www.linkedin.com/in/zhonglincao/?trk=public_profile_browsemap), [Rishikesh Magar*](https://www.linkedin.com/in/rishikesh-magar), [Yuyang Wang](https://yuyangw.github.io/), [Amir Barati Farimani](https://www.meche.engineering.cmu.edu/directory/bios/barati-farimani-amir.html) (*equal contribution) </br>
Carnegie Mellon University </br>

<img src="figs/pipeline.png" width="600">

This is the official implementation of ["MOFormer: Self-Supervised Transformer model for Metal-Organic Framework Property Prediction"](https://pubs.acs.org/doi/10.1021/jacs.2c11420). In this work, we propose a structure-agnostic deep learning method based on the Transformer model, named as <strong><em>MOFormer</em></strong>, for property predictions of MOFs. <strong><em>MOFormer</em></strong> takes a text string representation of MOF (MOFid) as input, thus circumventing the need of obtaining the 3D structure of a hypothetical MOF and accelerating the screening process. Furthermore, we introduce a self-supervised learning framework that pretrains the <strong><em>MOFormer</em></strong> via maximizing the cross-correlation between its structure-agnostic representations and structure-based representations of the crystal graph convolutional neural network (CGCNN) on >400k publicly available MOF data. Benchmarks show that pretraining improves the prediction accuracy of both models on various downstream prediction tasks. If you find our work useful in your research, please cite:

```
@article{doi:10.1021/jacs.2c11420,
    author = {Cao, Zhonglin and Magar, Rishikesh and Wang, Yuyang and Barati Farimani, Amir},
    title = {MOFormer: Self-Supervised Transformer Model for Metal–Organic Framework Property Prediction},
    journal = {Journal of the American Chemical Society},
    volume = {145},
    number = {5},
    pages = {2958-2967},
    year = {2023},
    doi = {10.1021/jacs.2c11420},
    URL = {https://doi.org/10.1021/jacs.2c11420}
}
```


## Getting Started

### Installation

Set up conda environment and clone the github repo

```
# create a new environment
$ conda create -n myenv python=3.9
$ conda activate moformer
$ conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
$ conda install --channel conda-forge pymatgen
$ pip install transformers
$ conda install -c conda-forge tensorboard

# clone the source code of MOFormer
$ git clone https://github.com/zcao0420/MOFormer
$ cd MOFormer
```

### Dataset

All the data used in this work can be found in `benchmark_datasets` folder. 

### Checkpoints

Pre-trained model can be found in `ckpt` folder. 

## Run the Model

### Pre-training

To pre-train the model using SSL from scratch, one can run `python pretrain_SSL.py`. The config file for the pretraining takes as input the directory of the cif file along with a file called `id_prop.npy`. The `id_prop.npy` has `cif id` and its corresponding `mof id` string representation. We have added folder named `cif_toy` with 100 cif files for MOFs and the the corresponding `id_prop.npy` for the data in the `cif_toy` folder. If you intend to run the pretraining for the `cif_toy` folder please make sure you update `config_multiview.yaml` indicating a proper location for the root directory.  
```
python pretrain_SSL.py
```

### Fine-tuning

To fine-tune the pre-trained Transformer, one can run `finetune_transformer.py` where the configurations are defined in `config_ft_transformer.yaml`. 
```
python finetune_transformer.py
```
Similarly, to fine-tune the pre-trained CGCNN, one can run `finetune_cgcnn.py` where the configurations are defined in `config_ft_cgcnn.yaml`.
```
python finetune_cgcnn.py
```

We also provide a jupyter notebook `demo.ipynb` for finetuning/supervised training.

## Acknowledgement
- CGCNN: [https://github.com/txie-93/cgcnn](https://github.com/txie-93/cgcnn)
- Crystal Twins: [https://github.com/RishikeshMagar/Crystal-Twins](https://github.com/RishikeshMagar/Crystal-Twins)
