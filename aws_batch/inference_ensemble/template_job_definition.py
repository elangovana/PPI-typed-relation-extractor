def get_job_definition(account, region, container_name, job_def_name, job_param_s3uri_destination, memoryInMB, ncpus,
                       role_name):
    """
This is the job definition for this sample job.
    :param account:
    :param region:
    :param container_name:
    :param job_def_name:
    :param memoryInMB:
    :param ncpus:
    :param role_name:
    :return:
    """
    return {
        "jobDefinitionName": job_def_name,
        "type": "container",
        # These are the arguments for the job
        "parameters": {
            "localpath": "/data",
            "s3destination": job_param_s3uri_destination,
            "s3src": job_param_s3uri_destination,
            "s3network_csv": job_param_s3uri_destination,
            "dataset": "PpiMulticlassDatasetFactory",
            "threshold": "0.0"
        },
        # Specify container & jobs properties include entry point and job args that are referred to in parameters
        "containerProperties": {
            "image": container_name,
            "vcpus": ncpus,
            "memory": memoryInMB,
            "command": [
                "bash",
                "scripts/ensemble_inference.sh",
                "Ref::s3src",
                "Ref::s3destination",
                "Ref::s3network_csv",
                "Ref::dataset",
                "Ref::localpath",
                "Ref::threshold"

            ],
            "jobRoleArn": "arn:aws:iam::{}:role/{}".format(account, role_name),
            "volumes": [
                {
                    "host": {
                        "sourcePath": "/dev/shm"
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
            "readonlyRootFilesystem": False,
            "privileged": True,
            "ulimits": [],
            "user": ""
            , "resourceRequirements": [
                {
                    "type": "GPU",
                    "value": "1"
                }
            ]
        },
        "retryStrategy": {
            "attempts": 5
        }
    }
