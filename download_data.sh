mkdir -p ./data

# Download Vocab/Model files
wget http://s3.amazonaws.com/proteindata/data_pytorch/pfam.model
wget http://s3.amazonaws.com/proteindata/data_pytorch/pfam.vocab

mv pfam.model data
mv pfam.vocab data

cd data
wget -c http://s3.amazonaws.com/proteindata/data_raw/secondary_structure.tar.gz -O - | tar -xz
# JSON for other tasks
# wget -c http://s3.amazonaws.com/proteindata/data_raw/proteinnet.tar.gz -O - | tar -xz
# wget -c http://s3.amazonaws.com/proteindata/data_raw/remote_homology.tar.gz -O - | tar -xz
# wget -c http://s3.amazonaws.com/proteindata/data_raw/fluorescence.tar.gz -O - | tar -xz
# wget -c http://s3.amazonaws.com/proteindata/data_raw/stability.tar.gz -O - | tar -xz

mkdir -p ./pickle