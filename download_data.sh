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
mkdir -p ./tf_records
cd tf_records
mkdir -p ./secondary_structure
# Download the TFRecord data for the Secondary Structure task
wget -c http://s3.amazonaws.com/proteindata/data/secondary_structure.tar.gz -O - | tar -xz
