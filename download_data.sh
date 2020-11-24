#!/bin/bash

#======================< DOWNLOAD SCRIPT >=====================#
#                                                              #
#   Pipeline script to download the relevent datasets.         #
#                                                              #
#               USAGE:    $ sh ./download_data.sh              #
#                                                              #
#==============================================================#

# Create new directory for the data
mkdir -p ./data
cd data
mkdir -p ./tf_record
cd tf_record
# Download the TFRecord data
wget -c http://s3.amazonaws.com/proteindata/data/secondary_structure.tar.gz -O - | tar -xz
# Download the JSON raw data
cd ..
mkdir -p ./json
cd json
wget -c http://s3.amazonaws.com/proteindata/data_raw/secondary_structure.tar.gz -O - | tar -xz
cd ../..
