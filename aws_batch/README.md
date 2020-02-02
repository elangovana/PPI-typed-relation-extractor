
# Register AWS batch job

## Prerequisites
1. Python 3.5+, https://www.python.org/downloads/release/python-350/ 
2. Install pip, see https://pip.pypa.io/en/stable/installing/ 
3. Install dependencies for this project
    ```bash
    pip install -r source/requirements.txt
    ``` 



## Training data prep
This is the full sequence to download the entire data..

### Download raw PPI function xml from Imex ftp

 1. Register aws batch
 
 ```bash
    export PYTHONPATH=./aws_batch

    python aws_batch/register_job_download_raw_files.py lanax/kegg-pathway-extractor:latest "<bucket>"

 ```
 
 
### Convert xml to json and filter relevant interaction types

 1. Register aws batch
 
 ```bash
    export PYTHONPATH=./aws_batch

    python aws_batch/register_job_dataprep_pipeline.py lanax/kegg-pathway-extractor:latest "<bucket>"

 ```
 To get specific interaction types
     
 ```bash
    export PYTHONPATH=./aws_batch

    python ./aws_batch/register_job_dataprep_pipeline.py lanax/kegg-pathway-extractor:latest aegovan-data "direct interaction,association,colocalization,phosphorylation,dephosphorylation,cleavage,enzymatic reaction,protein cleavage,methylation,ubiquitination,adp ribosylation,gtpase reaction,acetylation,covalent binding,deacetylation,demethylation,disulfide bond,atpase reaction,physical interaction,deubiquitination,hydroxylation,glycosylation,genetic interaction,putative self interaction,redox reaction,sumoylation,rna cleavage,self interaction,lipid cleavage,phosphotransfer,neddylation,palmitoylation,deamination,ampylation,demyristoylation,dna cleavage,transglutamination,deamidation,phospholipase reaction,deneddylation,depalmitoylation,dna elongation,isomerase reaction,proline isomerization  reaction"
 ```

## Running inference

### Convert gnormplus annotations into inference json formation 

1. Register aws batch

     ```bash
    export PYTHONPATH=./aws_batch

    python aws_batch/pubtator_annotations_inference_transformer_s3/register_job.py  lanax/kegg-pathway-extractor:latest s3://aegovan-data  --job-name ppi_pubmedinference_dataprep  --cpus 4   
    
    ```

2. Transform all data in s3, passing in the source & dest s3 files and the ncbi gene id mapping file
    ```bash
    export PYTHONPATH=./aws_batch

    python aws_batch/pubtator_annotations_inference_transformer_s3/submit_multiple_jobs.py  lanax/kegg-pathway-extractor:latest s3://aegovan-data  ppi_pubmedinference_dataprep  queue s3://aegovan-data/pubmed_json_parts_annotation_iseries/ s3://aegovan-data/pubmed_asbtract/inference_multi/ s3://aegovan-data/settings/HUMAN_9606_idmapping.dat
    
    ```

### Run large scale inference

 1. Register aws batch
 
    ```bash
    export PYTHONPATH=./aws_batch

    python aws_batch/inference_ensemble/register_job.py  11.dkr.ecr.us-east-2.amazonaws.com/ppi-extractor:inf-gpu-1.0.0-202001250025 s3://aegovan-data  --job-name kegg_inference_multi  --cpus 4   
    
    ```
    
 2. Submits files for inference
 
    ```bash
    export PYTHONPATH=./aws_batch

    python aws_batch/inference_ensemble/submit_multiple_jobs.py  kegg_inference_multi:6 gpu99  s3://aegovan-data/pubmed_asbtract/inference_multi/  s3://aegovan-data/pubmed_asbtract/predictions_multi/ s3://aegovan-data/results/ppi-bert-2019-11-24-17-25-37-406/output/model.tar.gz,s3://aegovan-data/results/ppi-bert-2019-11-24-17-23-34-503/output/model.tar.gz,s3://aegovan-data/results/ppi-bert-2019-11-24-17-22-16-517/output/model.tar.gz,s3://aegovan-data/results/ppi-bert-2019-11-24-17-18-42-192/output/model.tar.gz,s3://aegovan-data/results/ppi-bert-2019-11-24-17-16-59-176/output/model.tar.gz,s3://aegovan-data/results/ppi-bert-2019-11-24-17-15-51-079/output/model.tar.gz,s3://aegovan-data/results/ppi-bert-2019-11-24-17-14-21-187/output/model.tar.gz,s3://aegovan-data/results/ppi-bert-2019-11-24-17-11-07-931/output/model.tar.gz,s3://aegovan-data/results/ppi-bert-2019-11-24-17-12-37-190/output/model.tar.gz,s3://aegovan-data/results/ppi-bert-2019-11-24-17-09-56-491/output/model.tar.gz PpiMulticlassDatasetFactory --positives-filter-threshold 0.99
    
    ```