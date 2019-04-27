import argparse
import logging
import sys
from time import sleep

import boto3

from s3_utilities import list_files


def submit_job(job_def, queue_name, src_s3, dest_s3, s3_iddatfile, local_path="/data"):
    client = boto3.client('batch')
    response = client.submit_job(
        jobName='gnormplus_new',
        jobQueue=queue_name,
        jobDefinition=job_def,
        parameters={
            "localpath": local_path,
            "s3src": src_s3,
            "s3destination": dest_s3,
            "s3idmapping": s3_iddatfile

        },
        timeout={
            'attemptDurationSeconds': 86400 * 2
        }
    )

    print(response)


def submit_multiple(job_def, queue_name, s3_source_prefix, s3_destination_prefix, s3_idmappingdat,
                    local_path="/data"):
    # Submit job for each prefix
    for s3_bucket, s3_key in list_files(s3_source_prefix):
        sleep(1)
        submit_job(job_def, queue_name, "s3://{}/{}".format(s3_bucket, s3_key), s3_destination_prefix,
                   s3_idmappingdat, local_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)

    parser.add_argument("job_name",
                        help="Batch Job def name, e.g gnormplus:11")

    parser.add_argument("queue",
                        help="The name of the job queue", default="gnormplus")

    parser.add_argument("s3_source_prefix",
                        help="The s3 uri prefix that will contain the input/output data. e.g s3://mybucket/aws-batch-sample-python/")

    parser.add_argument("s3_dest_prefix",
                        help="The s3 uri prefix that will contain the input/output data. e.g s3://mybucket/aws-batch-sample-python/")

    parser.add_argument("s3_idmapping",
                        help="The s3 path for the idmapping dat file")

    args = parser.parse_args()

    # Register job
    submit_multiple(args.job_name, args.queue, args.s3_source_prefix, args.s3_dest_prefix, args.s3_idmapping)
    # submit_job(args.job_name, args.queue, args.s3_source_prefix, args.s3_dest_prefix, args.s3_idmapping)
    logger.info("Completed")
