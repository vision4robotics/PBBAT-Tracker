# PBBAT-tracker

| **Test passed**                                              |
| ------------------------------------------------------------ |
| [![matlab-2017b](https://img.shields.io/badge/matlab-2017b-yellow.svg)](https://www.mathworks.com/products/matlab.html) [![MatConvNet-1.0--beta25](https://img.shields.io/badge/MatConvNet-1.0--beta25%20-blue.svg)](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz) ![CUDA-8.0](https://img.shields.io/badge/CUDA-8.0-green.svg) |

> Matlab implementation of *Part-Based Background-Aware Tracking for UAV  with Convolutional Features* (PBBAT-tracker).

## Publication and Citation

This paper has been published by IEEE Access.

You can find this paper here: https://ieeexplore.ieee.org/document/8736485.

Please cite this paper as:

@ARTICLE{8736485,

  author={C. {Fu} and Y. {Zhang} and Z. {Huang} and R. {Duan} and Z. {Xie}},
  
  journal={IEEE Access}, 
  
  title={Part-Based Background-Aware Tracking for UAV With Convolutional Features}, 
  
  year={2019},
  
  volume={7},
  
  number={},
  
  pages={79997-80010},}

## Instructions

1. Download `imagenet-vgg-verydeep-19.mat` from [here](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) and put it in `./model`.
2. Download matconvnet toolbox [here](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz) and put it in `./external`.
3. Run `PBBAT_Demo.m`

Note: the original version is using CPU to run the whole program. If GPU version is required, just change `false` in the following lines in `run_PBBAT.m` to `true`:

```matlab
global enableGPU;
enableGPU = true;

vl_setupnn();
vl_compilenn('enableGpu', true);
```
## Acknowledgements
The parameter settings are partly borrowed from [BACF](http://www.hamedkiani.com/bacf.html) paper and convolutional feature extraction function is borrowed from [HCF](https://github.com/jbhuang0604/CF2).

## Results

![](./results/error.png "Precision plot")
![](./results/overlap.png "Success plot")
