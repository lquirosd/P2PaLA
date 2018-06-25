cBAD Dataset
============
[See](https://zenodo.org/record/).

The manuscript Oficio de Hipotecas de Girona (OHG) is pro-
vided by the Centre de Recerca d’Història Rural from the
Universitat de Girona (CRHR) V . This collection is composed
of hundreds of thousands of notarial deeds from the XVIII-
XIX century (1768-1862). Sales, redemption of censuses,
inheritance and matrimonial chapters are among the most
common documentary typologies in the collection. This
collection is divided in batches of 50 pages each, digitized
at 300ppi in 24 bit RGB color, available as TIF images
along with their respective ground-truth layout in PAGE
XML format, compiled by the HTR group of the PRHLT VI
center and CRHR. OHG pages are structured in a complex
layout composed of six different zone types, namely: $pag,
$tip, $par, $pac, $not, $nop;

> The dataset is divided randomly into training and test
set, 300 pages and 50 pages respectively. Experiments are
conducted on incremental subsets from 16 to 300 training
images, for Baseline detection only and integrated for 
baseline and zone segmentation .


Usage:
======
```bash
./run.sh
```

> - See [config](config_BL_only.txt) for details about training parameters for "Baselines detection only" experiments.
> - See [config](config_ZS_BL.txt) for details about training parameters for "Balelines + zone segmentation" experiments.

<!-- -->

> Dataset size is ~10.5GB, make sure to have at least 26GB of free disk space to store all the experiments (12 in total). 



Results
==================

## Baseline Experiments Metrics

|  Experiment  |  P   |  R   |  F1  | 
|:---------|:----:|:----:|:----:|
|<td colspan=3>BL only|
|  16    | 93.3 | 96.0 | 96.1 |
|  32    | 96.2 | 95.3 | 95.7 |
|  64    | 97.5 | 97.4 | 97.5 |
|  128   | 98.0 | 97.6 | 97.8 |
|  256   | 98.2 | 98.0 | 98.1 |
|  300   | 98.4 | 97.7 | 98.0 |
|<td colspan=3>BL + ZS|
|  16    | 81.4 | 92.5 | 86.2 |
|  32    | 79.6 | 95.0 | 86.0 |
|  64    | 91.8 | 95.8 | 93.4 |
|  128   | 94.8 | 96.4 | 95.5 |
|  256   | 93.3 | 96.4 | 94.4 |
|  300   | 96.2 | 97.1 | 96.5 |

## Zone Segmentation Metrics

| Experiment | Pixel Acc. | Mean Acc. | Mean IU | Freq. IU |
|:-----------|:----------:|:---------:|:-------:|:--------:|
| 16   | 81.7 | 68.4 | 58.9 | 77.2 |
| 32   | 81.1 | 68.5 | 57.8 | 74.7 |    
| 64   | 91.7 | 80.2 | 70.0 | 87.6 |    
| 128  | 95.1 | 84.5 | 76.0 | 91.8 |    
| 256  | 91.4 | 78.9 | 72.7 | 87.2 |    
| 300  | 92.4 | 86.9 | 76.3 | 88.7 |    

