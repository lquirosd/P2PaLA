cBAD Dataset
============
[See](https://zenodo.org/record/1297399).

This dataset arises from the READ project (Horizon 2020).

The dataset consists of a subset of documents from the Ratsprotokolle collection composed of minutes of the council meetings held from 1470 to 1805 (about 30.000 pages), which will be used in the READ project. This dataset is written in Early Modern German. The number of writers is unknown. Handwriting in this collection is complex enough to challenge the HTR software.

The training dataset is composed of 400 pages; most of the pages consist of a single block with many difficulties for line detection and extraction. The ground-truth in this set is in PAGE format and it is provided annotated at line level in the PAGE files.

> On this example "Training" and "Validation" splits are used as "train" and "test" respectively.


Usage:
======
```bash
./run.sh
```

> - See [config](config_BL_only.txt) for details about training parameters for "Baselines detection only" experiment.
> - See [config](config_zones_only.txt) for details about training parameters for "Zone segmentation only" experiment.
> - See [config](config_zone_BL.txt) for details about training parameters for "Balelines + zone segmentation" experiment.

<!-- -->

> Dataset size is ~500MB, make sure to have at least 6GB of free disk space to store all three experiments. 




Results
==================

## Baseline Experiments Metrics

|  Experiment  |  P   |  R   |  F1  | 
|:---------|:----:|:----:|:----:|
|  BL only   | 95.8 [92.7, 97.8] | 99.1 [98.6, 99.4] | 97.4 |
|  ZS + BL   | 94.4 | 98.9 | 96.58 |

## Zone Segmentation Metrics

| Experiment | Pixel Acc. | Mean Acc. | Mean IU | Freq. IU |
|:-----------|:----------:|:---------:|:-------:|:--------:|
| ZS only    | 95.8       | 93.3      | 82.7    | 91.3     |
| ZS + BL    | 95.5       | 91.4      | 84.5    | 91.6     |    

