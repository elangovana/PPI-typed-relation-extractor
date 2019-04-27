#!/bin/bash
src_s3=$1
dest_s3=$2
s3idmapping=$3
local_path=$4

set -e
set -x

tmp_dir=$(python -c 'import sys,uuid; sys.stdout.write(uuid.uuid4().hex)')

src_local_path=${local_path}/${tmp_dir}/input
iddat_local_path=${local_path}/${tmp_dir}/dat
iddat_local_file=${iddat_local_path}/$(echo ${s3idmapping} | rev | cut -d/ -f1 | rev)
dest_local_path=${local_path}/${tmp_dir}


mkdir -p ${src_local_path}
mkdir -p {${iddat_local_path}
mkdir -p ${dest_local_path}

# install aws s3
pip3 install awscli

# Copy data from s3
aws s3 cp ${src_s3} ${src_local_path}
aws s3 cp ${s3idmapping} ${iddat_local_path}

# Run
scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source_dir=${scripts_dir}/..
export PYTHONPATH=${source_dir}

python ${source_dir}/datatransformer/pubtator_annotations_inference_transformer.py ${src_local_path} ${dest_local_path} ${iddat_local_file}

# Copy results back s3
aws s3 cp --recursive ${dest_local_path}/ ${dest_s3}