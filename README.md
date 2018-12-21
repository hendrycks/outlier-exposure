# Outlier Exposure

This repository contains the essential code for the paper [_Deep Anomaly Detection with Outlier Exposure_](https://arxiv.org/abs/1812.04606) (ICLR 2019).

Requires Python 3+ and PyTorch 0.4.1+.

<img align="center" src="roc_curves.png" width="750">

## Overview

Outlier Exposure (OE) is a method for improving anomaly detection performance in deep learning models. Using an out-of-distribution dataset, we fine-tune a classifier so that the model learns heuristics to distinguish anomalies and in-distribution samples. Crucially, these heuristics generalize to new distributions. This repository contains a subset of the calibration and multiclass classification experiments. Please consult the paper for the full results and method descriptions.

Contained within this repository is code for the multiclass and calibration experiments for SVHN, CIFAR-10, CIFAR-100, and Tiny ImageNet.

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2019oe,
      title={Deep Anomaly Detection with Outlier Exposure},
      author={Hendrycks, Dan and Mazeika, Mantas and Dietterich, Thomas},
      journal={Proceedings of the International Conference on Learning Representations},
      year={2019}
    }

## Outlier Datasets

These experiments make use of numerous outlier datasets. Links for less common datasets are as follows, [Icons-50](https://github.com/hendrycks/robustness),
[Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz), and [Places365](http://places2.csail.mit.edu/download.html).

