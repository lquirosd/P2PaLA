#!/bin/bash 

#--- corpus will be available soon.

mkdir -p data/orig_data/
wget https://zenodo.org/record/<to_de_defined>/files/OHG.tar.gz -O data/orig_data/OHG.tar.gz
tar -xf data/orig_data/OHG.tar.gz -C data/orig_data
rm data/orig_data/OHG.tar.gz

c_dir=$PWD
#--- build test data structure
mkdir -p data/test/page
cd data/test
for f in $(<${c_dir}/test.lst); do
    ln -s $f .
    cd page
    dir=$(dirname $f)
    name=$(basename $f .tif)
    ln -s ${dir}/page/${name}.xml .
    cd ..
done
cd $c_dir
#--- build train data structure and train models
for b in {16,32,64,128,256,300}; do
    echo "working on ${b} ..."
    rm -rf data/${b}/train/*
    mkdir -p data/${b}/train/page
    cd data/${b}/train
    for f in $(head -n $b ${c_dir}/train.lst); do
        ln -s $f .
        cd page
        dir=$(dirname $f)
        name=$(basename $f .tif)
        ln -s ${dir}/page/${name}.xml .
        cd ..
    done
    cd $c_dir
    #--- do the actual train
    #--- ZS + BL model
    python /home/lquirosd/WORK/PhD/P2PaLA/P2PaLA.py --config config_ZS_BL.txt \
        --tr_data data/${b}/train \
        --te_data data/test \
        --work_dir work_${b} \
        --log_comment "_OHG_${b}"
    #--- Baselines only model
    python /home/lquirosd/WORK/PhD/P2PaLA/P2PaLA.py --config config_BL_only.txt \
        --tr_data data/${b}/train \
        --te_data data/test \
        --work_dir work_BL_${b} \
        --log_comment "_OHG_BL_${b}"
done

