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

## Run Docker

### Download and analyse the dataset with elastic search
#### Visualise
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

#### Prepare dataset


1. Download dataset from Imex ftp site ftp.ebi.ac.uk
    ```bash
    basedata=/home/ubuntu/data
 
    sudo docker run -v ${basedata}:/data  scripts/dowloadintactinteractions.sh /data  "<filepattern e.g. human*.xml>" "<optional s3 destination>"
    ```


## Run locally from source


1. Download dataset from Imex ftp site ftp.ebi.ac.uk
    ```bash
    cd ./source
    bash scripts/dowloadintactinteractions.sh "<localdir>" "<filepattern e.g. human*.xml>" "<optional s3 destination>"
    ```

2. Create a dataset locally from source
    
    ```bash
    export PYTHONPATH=./source
    python source/pipeline/main_pipeline_dataprep.py "<inputdir containing imex xml files>" "outputdir"
    ```
3. Create pubtator formatted abstracts so that GnormPlus can recognises entities

    ```bash
    export PYTHONPATH=./source
    python source/dataformatters/pubtatorAbstractOnlyFormatter.py "<datafilecreatedfrompreviousstep>" "<outputfile>"
    ```
4.  Extract entities using GNormPlus
    ```bash
    docker pull lanax/gnormplus
    docker run -it -v  lanax/gnormplus
    ```
    
    
4. Download NCBI to Uniprot Id mapping file
   
   From https://www.uniprot.org/downloads , download the ID mapping file ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/. This contains the ID mapping between UNIPROT and NCBI. We need this as GNormplus use NCBI gene id and we need the protein names.
   The dat file contains three columns, delimited by tab:
   
    - UniProtKB-AC 
    - ID_type 
    - ID
    
    e.g 
    ```text
    P43405	DNASU	6850
    P43405	GeneID	6850
    P43405	GenomeRNAi	6850
    A0A024R244	GeneID	6850
    A0A024R244	GenomeRNAi	6850
    A0A024R273	GeneID	6850
    A0A024R273	GenomeRNAi	6850
    A8K4G2	DNASU	6850
    ```
 
4. Download wordtovec pretrained models (either pubmed+pmc trained or  pubmed+pmc+wikipedia)and convert to text format 

    ```bash
    # Download word to vec trained only on pubmed and pmc
    wget  wget -O /data/PubMed-and-PMC-w2v.bin http://evexdb.org/pmresources/vec-space-models/PubMed-and-PMC-w2v.bin
 
    python ./source/dataformatters/main_wordToVecBinToText.py /data/PubMed-and-PMC-w2v.bin /data/PubMed-and-PMC-w2v.bin.txt
    ```
    
    ```bash
    # Download word to vec trained only on pubmed and pmc and wikipedia
    wget  wget -O /data/PubMed-and-PMC-w2v.bin http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin
 
    python ./source/dataformatters/main_wordToVecBinToText.py /data/wikipedia-pubmed-and-PMC-w2v.bin /data/wikipedia-pubmed-and-PMC-w2v.bin.txt
    ```

4. Run train job
    ```bash
    export PYTHONPATH=./source
    python source/algorithms/main_train.py Linear train.json val.json ./tests/test_algorithms/sample_PubMed-and-PMC-w2v.bin.txt 200 outdir
    ```

5. Consolidated train + predict
    ```bash
     #!/usr/bin/env bash
  
     # Init
     # Type of network, Linear is MLP, Cnn is Cnn, CnnPos is with position embedding
     network=CnnPos
     base_dir=/data
     s3_dest=s3://yourbucket/results
    
    
     fmtdt=`date +"%y%m%d_%H%M"`
     base_name=model_${network}_${fmtdt}
     outdir=${base_dir}/${base_name}
     echo ${outdir}
     mkdir ${outdir}
      
     export PYTHONPATH="./source"
     
     mkdir ${outdir}
     
     # Git head to trace to source to reproduce run
     git log -1 > ${outdir}/run.log
     
     # Train
     python ./source/algorithms/main_train.py ${network}  /data/train_unique_pub_v3_lessnegatve.json /data/val_unique_pub_v3_lessnegatve.json /data/wikipedia-pubmed-and-PMC-w2v.bin.txt 200 ${outdir}  --epochs 50  --log-level INFO >> ${outdir}/run.log 2>&1
     
     # Predict on validation set
     python ./source/algorithms/main_predict.py ${network}  /data/test_unique_pub_v3_lessnegatve.json ${outdir}  ${outdir} >> ${outdir}/run.log 2>&1
     mv ${outdir}/predicted.json ${outdir}/predicted_test_unique_pub_v3_lessnegatve.json
     
     # Predict on test set
     python ./source/algorithms/main_predict.py ${network}  /data/val_unique_pub_v3_lessnegatve.json ${outdir}  ${outdir} >> ${outdir}/run.log 2>&1
     mv ${outdir}/predicted.json ${outdir}/predicted_val_unique_pub_v3_lessnegatve.json
    
     #Copy results to s3
     aws s3 sync ${outdir} ${s3_dest}/${base_name} >> ${outdir}/synclog 2>&1
    
    ```