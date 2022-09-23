#!/bin/bash

####################################
#   GET PTB-XL DATABASE
####################################

mkdir -p data
cd data || return
wget https://storage.googleapis.com/ptb-xl-1.0.1.physionet.org/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip --no-check-certificate
unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
mv ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1 ptbxl
cd ..