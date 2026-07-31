#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gs

mkdir -p data

python -c "import kagglehub, shutil;path = kagglehub.dataset_download('arenagrenade/llff-dataset-full');shutil.copytree(path, 'data', dirs_exist_ok=True);shutil.rmtree(path)"

unzip pcd.zip
cp -r pcd/* data/nerf_llff_data/
rm -r pcd.zip pcd