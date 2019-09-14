ARG REGION=us-west-2
ARG DEVICE=cpu
ARG AWS_PYTORCH_SAGEMAKER_ECR_ACCOUNT=520713654638
FROM ${AWS_PYTORCH_SAGEMAKER_ECR_ACCOUNT}.dkr.ecr.$REGION.amazonaws.com/sagemaker-pytorch:1.0.0-${DEVICE}-py3

# NOTE: Make sure /opt/ml/code is empty,
# Else Sagemaker container framework will NOT download source code from s3 to this directory

COPY ./src/requirements_dev.txt  /tmp

RUN pip install -r /tmp/requirements_dev.txt && rm /tmp/requirements_dev.txt




