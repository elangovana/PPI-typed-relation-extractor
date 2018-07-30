#!/bin/bash
s3destination=$1
set -e
mkdir data
mkdir outdir

#Download ftp files
scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
. $scripts_dir/dowloadintactinteractions.sh data

#process data
source_dir=$scripts_dir/..
export PYTHONPATH=$source_dir
python $source_dir/dataloader/bulkImexDataPreprocessor.py data outdir



if ["$s3destination" == ""]; then
    echo "No s3destination argument passed, hence not copying to s3. If you want to copy data to s3 .."
    echo "$0 s3://mybucket/data"
else

    aws s3 copy outdir/* $s3destination
fi
