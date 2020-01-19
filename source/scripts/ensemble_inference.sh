#!/bin/bash
src_s3=$1
dest_s3=$2
network_s3_csv=$3
dataset=$4
local_path=$5
threshold=$6

set -e
set -x

tmp_dir=$(python -c 'import sys,uuid; sys.stdout.write(uuid.uuid4().hex)')

src_local_path=${local_path}/${tmp_dir}/input
src_file_name=$(echo ${src_s3} | rev | cut -d/ -f1 | rev)
src_local_file=${src_local_path}/${src_file_name}

network_local_path=${local_path}/${tmp_dir}/model_artefacts
dest_local_path=${local_path}/${tmp_dir}/output


mkdir -p ${src_local_path}
mkdir -p ${network_local_path}
mkdir -p ${dest_local_path}

# install aws s3
pip3 install awscli

# Copy data from s3
aws s3 cp ${src_s3} ${src_local_path}

# Run
scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source_dir=${scripts_dir}/..
export PYTHONPATH=${source_dir}

python ${source_dir}/algorithms/ensemble_inference_batchs3.py --input-comma-sep-path  ${network_s3_csv} --dest-dir ${network_local_path}
python ${source_dir}/algorithms/main_predict.py ${dataset}  ${src_local_file}  ${network_local_path}   ${dest_local_path}  --positives-filter-threshold ${threshold}
mv ${dest_local_path}/predicted.json ${dest_local_path}/${src_file_name}.prediction.json

# Copy results back s3
aws s3 cp --recursive ${dest_local_path}/ ${dest_s3}

# Clean up
rm -rf ${dest_local_path}
rm -rf ${src_local_path}
rm -rf ${network_local_path}