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
export AWS_ACCOUNT=''
export AWS_REGION=''
export tag='201809170901'
 python aws_batch/register_job.py $AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/awscomprehend-sentiment-demo:$tag 
```