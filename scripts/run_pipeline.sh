#!/bin/bash
datadir=$1
filepattern=$2
s3destination=$3
processeddatadir=$datadir/processed
set -e
mkdir $processeddatadir


#Download ftp files
echo "Downloading files.."
scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
bash $scripts_dir/dowloadintactinteractions.sh $datadir $filepattern

#process data
echo "Processing data..in directory $datadir"
source_dir=$scripts_dir/..
export PYTHONPATH=$source_dir
python $source_dir/dataloader/bulkImexDataPreprocessor.py $datadir $processeddatadir


echo "Attempting to copy to s3 if specified $s3destination"
if [ "$s3destination" == "" ]; then
    echo "No s3destination argument passed, hence not copying to s3. If you want to copy data to s3 .."
    echo "$0  <outdir>  <filepattern> <s3://mydesbucket/data>"
else
    echo "Copying from $processeddatadir/* to $s3destination"
    aws s3 cp $processeddatadir/* $s3destination --recursive
fi
