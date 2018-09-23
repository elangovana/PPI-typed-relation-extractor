import argparse
import boto3


class RegisterJob:

    def __init__(self, client=None, account=None, aws_region=None):
        self.client = client or boto3.client('batch')
        self.account = account or boto3.client('sts').get_caller_identity().get('Account')
        self.region = aws_region or boto3.session.Session().region_name

    def run(self, container_name: str):
        """
        Registers a job with aws batch.
        :param container_name: The name of the container to use e.g 324346001917.dkr.ecr.us-east-2.amazonaws.com/awscomprehend-sentiment-demo:latest
        """

        job_definition = {
            "jobDefinitionName": "KeggPathWayExtractor_downloadfile",
            "type": "container",
            "parameters": {
                "file_pattern": "human_*.xml",
                "s3_destination": "s3://<bucker>/prefix/",
                "local_dir": "/data"
            },
            "containerProperties": {
                "image": container_name,
                "vcpus": 1,
                "memory": 2000,
                "command": [
                    "scripts/dowloadintactinteractions.sh",
                    "Ref::local_dir",
                    "Ref::file_pattern",
                    "Ref::s3_destination"
                ],
                "jobRoleArn": "arn:aws:iam::{}:role/Batch".format(self.account),
                "volumes": [
                    {
                        "host": {
                            "sourcePath": "KeggPathWayExtractor_downloadfile"
                        },
                        "name": "data"
                    }
                ],
                "environment": [
                    {
                        "name": "AWS_DEFAULT_REGION",
                        "value": self.region
                    }
                ],
                "mountPoints": [
                    {
                        "containerPath": "/data",
                        "readOnly": False,
                        "sourceVolume": "data"
                    }
                ],
                "readonlyRootFilesystem": False,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("containerimage",
                        help="Container image, e.g docker pull lanax/kegg-pathway-extractor:latest")

    job = RegisterJob()
    args = parser.parse_args()
    result = job.run(args.containerimage)
    print(result)
