Installation
============

Installation instructions are provided for linux and under [conda](https://conda.io/docs/user-guide/install/index.html) 
enviroment. We recomend to create a new conda environment for this installation. 

P2PaLA is implemented in [PyTorch](http://pytorch.org/), and depends on the following:

- [CUDA >=8.0](https://developer.nvidia.com/cuda-downloads)
- [cuDNN >= 6.0](https://developer.nvidia.com/cudnn)

> Note that currently we only support GPU. You need to use NVIDIA's cuDNN library.
> Register first for the CUDA Developer Program (it's free) and download the library
> from [NVIDIA's website](https://developer.nvidia.com/cudnn).

> This code is tested only on Python2.7, Python3. should work but untested.

Once PyTorch is installed the following libs are required:

- [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) (v0.9).
- [OpenCv](https://github.com/opencv/opencv/releases/tag/3.3.1) (3.1.0).

Then execute:
```bash
git clone https://github.com/lquirosd/P2PaLA.git
```

Return to [docs](./README.md) @-----@ See Next: [Usage](./USAGE.md)
