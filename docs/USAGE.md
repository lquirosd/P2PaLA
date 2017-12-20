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
│   ├── page
│   │   ├── prod_0.xml
│   │   └── prod_1.xml
│   ├── prod_0.jpg
│   └── prod_1.jpg
├── test
│   ├── page
│   │   ├── test_0.xml
│   │   └── test_1.xml
│   ├── test_0.jpg
│   └── test_1.jpg
├── train
│   ├── page
│   │   ├── train_0.xml
│   │   └── train_1.xml
│   ├── train_0.jpg
│   └── train_1.jpg
└── val
    ├── page
    │   ├── val_0.xml
    │   └── val_1.xml
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
5. For detail about arguments and config file, see the [full help](./help.md) or `python P2PaLa.py -h`. 
6. For more detailed example see [egs](../egs):
    * cBAD complex competition dataset [see](../egs/cBAD_complex)

Return to [docs](./README.md)
