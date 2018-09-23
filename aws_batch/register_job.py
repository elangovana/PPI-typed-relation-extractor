import argparse
import boto3


class RegisterJob:

    def __init__(self, client = None):
        self.client = client or boto3.client('batch')

    def run(self, container_name: str):
        """
        Registers a job with aws batch.
        :param container_name: The name of the container to use e.g 324346001917.dkr.ecr.us-east-2.amazonaws.com/awscomprehend-sentiment-demo:latest
        """
        account_id=container_name.split(".")[0]
        region = container_name.split(".")[3]
        job_definition = {
            "jobDefinitionName": "KeggPathWayExtractor",
            "type": "container",
            "parameters": {
                "mode": "single",
                "text": "This is awesome"
            },
            "containerProperties": {
                "image": container_name,
                "vcpus": 1,
                "memory": 2000,
                "command": [
                    "Ref::mode",
                    "Ref::text"
                ],
                "jobRoleArn": "arn:aws:iam::{}:role/Batch".format(account_id),
                "volumes": [
                    {
                        "host": {
                            "sourcePath": "DocumentSentimentAnalysisdata"
                        },
                        "name": "data"
                    }
                ],
                "environment": [
                    {
                        "name": "AWS_DEFAULT_REGION",
                        "value": region
                    }
                ],
                "mountPoints": [
                    {
                        "containerPath": "/data",
                        "readOnly": False,
                        "sourceVolume": "data"
                    }
                ],
                "readonlyRootFilesystem": True,
                "privileged": True,
                "ulimits": [],
                "user": ""
            },
            "retryStrategy": {
                "attempts": 1
            }
        }

        reponse = self.client.register_job_definition(**job_definition)
        return reponse



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("containerimage",
                        help="Container image, e.g docker pull lanax/kegg-pathway-extractor:latest")

    job = RegisterJob()
    args = parser.parse_args()
    result = job.run(args.containerimage)
    print(result)
