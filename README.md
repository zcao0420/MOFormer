# MOFormer

<b>MOFormer: Self-Supervised Transformer model for Metal-Organic Framework Property Prediction</b> </br>
[[arXiv]](https://arxiv.org/abs/2210.14188) [[PDF]](https://arxiv.org/pdf/2210.14188.pdf) </br>
[Zhonglin Cao*](https://www.linkedin.com/in/zhonglincao/?trk=public_profile_browsemap), [Rishikesh Magar*](https://www.linkedin.com/in/rishikesh-magar), [Yuyang Wang](https://yuyangw.github.io/), [Amir Barati Farimani](https://www.meche.engineering.cmu.edu/directory/bios/barati-farimani-amir.html) </br>
Carnegie Mellon University </br>

<img src="figs/pipeline.png" width="600">

This is the official implementation of ["MOFormer: Self-Supervised Transformer model for Metal-Organic Framework Property Prediction"](https://arxiv.org/abs/2210.14188). If you find our work useful in your research, please cite:

```
@article{cao2022moformer,
    title={MOFormer: Self-Supervised Transformer model for Metal-Organic Framework Property Prediction},
    author={Cao, Zhonglin and Magar, Rishikesh and Wang, Yuyang and Barati Farimani, Amir},
    journal={arXiv preprint arXiv:2210.14188},
    year={2022}
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

To pre-train the model from scratch, one can run 
```
python pretrain.py
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