P2PaLA
======

Page to [PAGE](http://www.primaresearch.org/tools/PAGELibraries) Layout Analysis (P2PaLA) is a toolkit for Document Layout Analysis based on Neural Networks.

If you find this toolkit useful in your research, please cite:
```
@misc{p2pala2017,
  author = {Lorenzo Quirós},
  title = {P2PaLA: Page to PAGE Layout Analysis tookit},
  year = {2017},
  publisher = {GitHub},
  note = {GitHub repository},
  howpublished = {\url{https://github.com/lquirosd/P2PaLA}},
}
```

Requirements
===========

- Linux (OSX may work, but untested.).
- [PyTorch](http://pytorch.org) (0.2.0\_4).
- [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) (v0.9).
- [OpenCv](https://github.com/opencv/opencv/releases/tag/3.3.1) (3.1.0).
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested).

Usage
=====
1. Input data must follow the folder structure `data_tag/page`, where images must be into the `data_tag` folder and xml files into `page`. For example:
```bash
mkdir -p data/{train,val,test,prod}/page;
tree data;
```
```
data
├── prod
│   ├── page
│   │   ├── prod_0.xml
│   │   └── prod_1.xml
│   ├── prod_0.jpg
│   └── prod_1.jpg
├── test
│   ├── page
│   │   ├── test_0.xml
│   │   └── test_1.xml
│   ├── test_0.jpg
│   └── test_1.jpg
├── train
│   ├── page
│   │   ├── train_0.xml
│   │   └── train_1.xml
│   ├── train_0.jpg
│   └── train_1.jpg
└── val
    ├── page
    │   ├── val_0.xml
    │   └── val_1.xml
    ├── val_0.jpg
    └── val_1.jpg
```
2. Run the tool.
```bash
python P2PaLA.py --config config.txt --tr_data ./data/train --te_data ./data/test --log_comment "_foo"
```
3. Use TensorBoard to visualize train status:
```bash
tensorboard --logdir ./work/runs
```
4. xml-PAGE files must be at "./work/results/test/"
> We recomend [Transkribus](https://transkribus.eu/Transkribus/) or [nw-page-editor](https://github.com/mauvilsa/nw-page-editor) 
> to visualize and edit PAGE-xml files.
5. For detail about arguments and config file, see [docs](docs) or `python P2PaLa.py -h`. 
6. For more detailed example see [egs](egs)


License
=======
GNU General Public License v3.0
See [LICENSE](LICENSE) to see the full text.

Acknowledgments
===============
Code is inspired by [pix2pix](https://github.com/phillipi/pix2pix) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

To-do
=====
- [ ] Providen and example of use
- [ ] Provide Docker 
- [ ] Include [BaselinePage](https://github.com/PRHLT/BaseLinePage) to detect baselines.
- [ ] Test on Mac/OS.
- [ ] Test on CPU.
