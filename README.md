# A Comparative Study of Self-Supervised Methods for Few-Shot Learning

This repo contains a testbed for an empirical comparison of classical few-shot methods and self-supervised methods on few-shot classification tasks.

## Enviroment
 - python3
 - [pytorch](http://pytorch.org/) 1.0.1
 - torchvision 0.2.1
 - pillow 6.1.0

## Getting started
### Omniglot
* Change directory to `./data/omniglot`
* run `./download_omniglot.sh` 

### Omniglot with class augmentation
* Change directory to `./data/omniglot`
* run `./download_omniglot_aug.sh`

### miniImageNet
* Change directory to `./data/miniImagenet`
* run `./download_miniImagenet.sh` 

(WARNING: The script downloads the 155G ImageNet dataset. If you already have the data, you can comment out lines 5-6 in `download_miniImagenet.sh`.) 

## Train and test
Run `python ./main.py --dataset [DATASETNAME] --model [MODELNAME] --backbone [BACKBONE] [--OPTIONARG]`

For example, run `python ./main.py --dataset omniglot_aug --model ISIF_M --backbone Conv4 --z_dim 64`
For more options check `python ./main.py --help`

## Results
* The results will be stored in the folder specified by `--ckpt_dir`
* If `--eval` is set to 'val' or 'test', this will create a file log_val.txt or log_test.txt, respectively, which contains the performance on downstream tasks.

## Disclaimer
This repository builds upon multiple publicly available codebases. In particular we have used and modified code from the following projects:

* Dataset preparation
https://github.com/wyharveychen/CloserLookFewShot
* Prototypical Networks
https://github.com/jakesnell/prototypical-networks
* RotNet
https://github.com/gidariss/FeatureLearningRotNet
* AET (Autoencoding transformations)
https://github.com/maple-research-lab/AET
* NPID (Non-parametric instance discrimination)
https://github.com/zhirongw/lemniscate.pytorch
* ISIF (Invariant and Spreading Instance Feature)
https://github.com/mangye16/Unsupervised_Embedding_Learning