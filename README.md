# Assignment 1 

## Exercise 1: Fine-tune classification model using MMClassification

In this exercise, I completed the fine-tune training based on the pre-training model provided by MMClassification.

### Overview

I fine-tuned a ResNet-50 pretrained model using data in the ImageNet format, with 5 classes in total.

Some of my main modified file format is:
```
mmpretrain/
│
├── configs/
|   ├── resnet
|   |   ├── resnet50_8xb32-ft_custom.py
|   ├── 
│   └── resnet50_8xb32_in1k_20210831-ea4938fc.pth
│
├── flower_dataset/
│   ├── train/
│   ├── val/
│   ├── classes.txt
|   ├── train.txt
|   └── val.txt
|
└── mmpretrain/
    └── datasets/
        ├── categories.py
        └── imagenet.py
```
The dataset is organized into ImageNet format in the document flower_dataset.

The configuration file is located in [mmpretrain/configs/resnet/resnet50_8xb32-ft_custom.py](mmpretrain/configs/resnet/resnet50_8xb32-ft_custom.py)

### Environment
Run these commands in sequence to set up the environment.\
Please do not point the command line to this folder when setting up the environment.\
`git clone https://github.com/open-mmlab/mmpretrain.git`\
`cd mmpretrain`\
`pip install -U openmim && mim install -e .`

### Train and Test
Run this commands to fine-tune the ResNet-50 pre-trained model.

`python tools/train.py configs/resnet/resnet50_8xb32-ft_custom.py`

The trained model is stored in [mmpretrain/work_dirs/resnet50_8xb32-ft_custom](mmpretrain/work_dirs/resnet50_8xb32-ft_custom) after training.

Since there are only five types, only the top-1 accuracy is calculated in the evaluation criteria. The top-1 accuracy can reach approximately 97%.
