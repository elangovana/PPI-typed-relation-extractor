ARG REGION=us-west-2
ARG DEVICE=cpu
ARG AWS_PYTORCH_SAGEMAKER_ECR_ACCOUNT=520713654638
FROM ${AWS_PYTORCH_SAGEMAKER_ECR_ACCOUNT}.dkr.ecr.$REGION.amazonaws.com/sagemaker-pytorch:1.0.0-${DEVICE}-py3

# NOTE: Make sure /opt/ml/code is empty,
COPY ./source  /opt/ml/code

# Else Sagemaker container framework will NOT download source code from s3 to this directory
COPY ./source/requirements_prod.txt  /tmp


ENV PATH=/opt/ml/code:/opt/ml/code/algorithms:${PATH}

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code


RUN pip install -r /tmp/requirements_prod.txt && rm /tmp/requirements_prod.txt



