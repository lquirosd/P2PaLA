#!/bin/bash
#--- Prepare folder
mkdir -p data/{train,test}/page

#--- download original data
wget https://zenodo.org/record/1297399/files/Train-And-Val-ICFHR-2016.tgz -O data/PublicData.tgz
#--- decompress files
tar -xf data/PublicData.tgz -C data/


mv data/PublicData/Training/Images/* data/train/.
mv data/PublicData/Training/page/page/*.xml data/train/page/.
mv data/PublicData/Validation/Images/* data/test/.
mv data/PublicData/Validation/page/page/*.xml data/test/page/.

#rm data/PublicData.tgz
#rm -rf data/PublicData

#--- run train
python ../../P2PaLA.py --config config_BL_only.txt --log_comment "_Bozen_BL_only"
python ../../P2PaLA.py --config config_zones_only.txt --log_comment "_Bozen_zones_only"
python ../../P2PaLA.py --config config_zones_BL.txt --log_comment "_Bozen_ZS_BL"
