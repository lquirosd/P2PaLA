#!/bin/bash
#--- Prepare folder
mkdir -p data/{train,test}/page
mkdir -p data/orig_data
#--- download original data and rename becouse the espaces and stuff
wget ftp://scruffy.caa.tuwien.ac.at/staff/read/cBAD-ICDAR2017/Train%20-%20Baseline%20Competition%20-%20Complex%20Documents.zip -O data/orig_data/train_complex.zip
wget ftp://scruffy.caa.tuwien.ac.at/staff/read/cBAD-ICDAR2017/Test%20-%20Baseline%20Competition%20-%20Complex%20Documents.zip -O data/orig_data/test_complex.zip
#--- unzip files
unzip data/orig_data/train_complex.zip -d data/orig_data
mv data/orig_data/Baseline\ Competition\ -\ Complex\ Documents/ data/orig_data/train_complex
unzip data/orig_data/test_complex.zip -d data/orig_data
mv data/orig_data/Baseline\ Competition\ -\ Complex\ Documents/ data/orig_data/test_complex

rm data/orig_data/train_complex.zip
rm data/orig_data/test_complex.zip
#--- Due the name insonsistencies across the diferent folders all files have been
#--- renamed using standard "cBAD_complex_[train,test]_<#>.[xml,jpg,...]
#--- all the changes are stored on "file_mapping.lst" file

#--- move badplaced xml files
mv data/orig_data/test_complex/ABP_FirstTestCollection/*.xml data/orig_data/test_complex/ABP_FirstTestCollection/page/.
mv data/orig_data/test_complex/EPFL_VTM_FirstTestCollection/*.xml data/orig_data/test_complex/EPFL_VTM_FirstTestCollection/page/.

#--- 1) Build file_mapping.lst
ORIG=../orig_data/train_complex
id=0
echo -n "" > data/orig_data/train_file_mapping.lst
for f in `find data/orig_data/train_complex -iname *.jpg`; do 
    d=`dirname $f`; 
    folder=`basename $d`; 
    name=`basename $f`; 
    name="${name%.*}"; 
    echo "cBAD_complex_train_$id $folder $name" >> data/orig_data/train_file_mapping.lst;
    id=$(( id + 1 ));
done
#--- 1) map original images to new file names 
cd data/train
while read p; do 
    fields=( $(echo $p | cut -d ' ' -f1- ));
    ln -s $( ls ${ORIG}/"${fields[1]}"/"${fields[2]}".*) "${fields[0]}".jpg
done < ../orig_data/train_file_mapping.lst
#--- page as well
cd page
while read p; do 
    fields=( $(echo $p | cut -d ' ' -f1- ));
    ln -s $( ls ../${ORIG}/"${fields[1]}"/page/"${fields[2]}".xml) "${fields[0]}".xml
done < ../../orig_data/train_file_mapping.lst
cd ../../..
ORIG=../orig_data/test_complex
id=0
echo -n "" > data/orig_data/test_file_mapping.lst
for f in `find data/orig_data/test_complex -iname *.jpg`; do 
    d=`dirname $f`; 
    folder=`basename $d`; 
    name=`basename $f`; 
    name="${name%.*}"; 
    echo "cBAD_complex_train_$id $folder $name" >> data/orig_data/test_file_mapping.lst;
    id=$(( id + 1 ));
done
#--- 1) map original images to new file names 
cd data/test/
while read p; do 
    fields=( $(echo $p | cut -d ' ' -f1- ));
    ln -s $( ls ${ORIG}/"${fields[1]}"/"${fields[2]}".*) "${fields[0]}".jpg
done < ../orig_data/test_file_mapping.lst
cd ../../
#--- run train
python ../../P2PaLA.py --config config.txt --log_comment "_cBAD_corpus"

cd work/results/prod/page
mkdir cBADresults
while read p; do 
    fields=( $(echo $p | cut -d ' ' -f1- )); 
    mkdir -p cBADresults/"${fields[1]}"; 
    cp "${fields[0]}".xml cBADresults/"${fields[1]}"/"${fields[2]}".xml; 
done < ../../../../data/orig_data/test_file_mapping.lst
cd cBADresults
tar -cvf cBAD_complex_results.tar.gz *

echo "Now you can check your results on Scripnet"
echo "https://scriptnet.iit.demokritos.gr/competitions/5/1/2/"
echo `realpath cBAD_complex_results.tar.gz`
cd ../../../../..
