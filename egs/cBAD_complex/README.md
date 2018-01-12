cBAD Dataset
============
[See](https://scriptnet.iit.demokritos.gr/competitions/5/1/).

The database of Track A [Simple Documents] consists of 755 images extracted from 9 different archival collections. The dataset comprises images with additional PAGE XMLs [1](http://www.primaresearch.org/tools). The PAGE XML contains text regions, e.g. paragraphs. Thus a layout analysis or text detection needs not to be performed on this dataset. Only handwritten text is present and the dataset contains no tables. The groundtruth of the test-set will be released after evaluating all submitted methods and the final results being made public.

Track B [Complex Documents] contains mixed documents. Though most documents are handwritten, printed documents, book covers, empty pages, and tables are contained in this track. While Track A has locally skewed text-lines, text-lines in Track B are rotated up to 180°

> On this example only Complex Track is used.


Usage:
======
```bash
./run.sh
```

> Dataset size more then 2GB, make sure to have at least 6GB of free disk space to store all the experiment. 

> See [config](config.txt) for details about training parameters.

ICDAR 2017 Results
==================

## Complex Track

Following table shows results published on [ICDAR 2017](http://u-pat.org/ICDAR2017/index.php) proceddings 
plus the results of this experiment (P2PaLA row).

|  Method  |  P   |  R   |  F1  | 
|:---------|:----:|:----:|:----:|
|  DMRZ    | **85.4** | **86.3** | **85.9** |
|  **P2PaLA**  | 83.25 | 85.73 | 84.47 |
|  BYU     | 77.3 | 82.0 | 79.6 |
|  IRISA   | 69.2 | 77.2 | 73.0 |
|  UPVLC   | 83.3 | 60.6 | 70.2 |

As you can notice, results are pretty close to competition winner. Although no 
hyperparameter tunning is performed.

Corpus Notes
============

## Complex Track

### Train data

* number of pages: 270
* color schema: 60 Gray, 210 sRGB
* size: 209 different sizes, from 1504x1194 to 7456x6104
* orientation: both Portrait and landscape
* Baselines:
  - total: 21684
  - average per page: 80.3
  - min: 0
  - max: 472
  - [histogram](./imgs/train_data_baseline_histogram.png)

### Test data

* number of pages: 1010
* color schema: 163 Gray, 847 sRGB
* size: 678 different sizes, from 982x3127 to 7472x6088
* orientation: both Portrait and landscape
* Baselines: **blind test**
