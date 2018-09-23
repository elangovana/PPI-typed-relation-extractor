# Register AWS batch job

## Prerequisites
1. Python 3.5+, https://www.python.org/downloads/release/python-350/ 
2. Install pip, see https://pip.pypa.io/en/stable/installing/ 
3. Install dependencies for this project
```bash
pip install -r source/requirements.txt
``` 



## How to run
```bash
export tag='latest'
python aws_batch/register_job_download_raw_files.py lanax/kegg-pathway-extractor:$tag 
```