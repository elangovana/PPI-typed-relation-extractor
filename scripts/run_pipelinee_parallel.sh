#!/usr/bin/env bash

region=$1
esdomain=$2
accesskey=$3
accesssecret=$4
s3path=$5

basedata=/home/ubuntu/data

for i in `seq 0 9`;
do
        echo $i
        sudo mkdir ${basedata}_$i
        sudo docker run -v ${basedata}_$i:/data --env elasticsearch_domain_name=$esdomain --env AWS_ACCESS_KEY_ID=$accesskey   --env AWS_REGION=$region --env AWS_SECRET_ACCESS_KEY=$accesssecret lanax/kegg-pathway-extractor:latest /data human_$i*.xml $s3path > human_$1.log 2>&1 &
done
