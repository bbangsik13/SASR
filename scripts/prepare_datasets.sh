#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate sasr

mkdir -p data

# llff
python -c "import kagglehub, shutil;path = kagglehub.dataset_download('arenagrenade/llff-dataset-full');shutil.copytree(path, 'data', dirs_exist_ok=True);shutil.rmtree(path)"

# mipnerf360
wget -P data/ http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip data/360_v2.zip -d ./data/mipnerf360/
rm data/360_v2.zip data/mipnerf360/*txt

# mvs
unzip  pcd.zip -d ./data/
rm pcd.zip