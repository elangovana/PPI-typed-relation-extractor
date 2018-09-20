[![Build Status](https://travis-ci.org/elangovana/kegg-pathway-extractor.svg?branch=master)](https://travis-ci.org/elangovana/kegg-pathway-extractor)

# kegg-pathway-extractor
Given a Kegg Pathway Id, e.g path:ko05215, extracts protein interactions defined in that pathway and the type of interaction.

# Prerequisite
1. Data
   Get the MIPS Interaction XML file & extract
   ```shell
   wget http://mips.helmholtz-muenchen.de/proj/ppi/data/mppi.gz
   gunzip mppi.gz 
   ```  
2. Download pretrained word embeddings
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

# Run Docker
1. Sample download intact files with pattern human0* and elastic search index
```bash
region=$1
esdomain=$2
accesskey=$3
accesssecret=$4
s3path=$5

basedata=/home/ubuntu/data
file_pattern=human0*

script=scripts/run_pipeline_download_esindex.sh

sudo docker run -v ${basedata}:/data --env elasticsearch_domain_name=$esdomain --env AWS_ACCESS_KEY_ID=$accesskey   --env AWS_REGION=$region --env AWS_SECRET_ACCESS_KEY=$accesssecret lanax/kegg-pathway-extractor:latest $script /data $file_pattern $s3path 
```
