import argparse
import logging
import sys
from time import sleep

import boto3

from s3_utilities import list_files


def submit_job(job_def, queue_name, s3_source_prefix, dest_s3, s3_network_prefix_csv, network_type,
               positives_filter_threshold,
               local_path="/data"):
    client = boto3.client('batch')
    response = client.submit_job(
        jobName='Inference',
        jobQueue=queue_name,
        jobDefinition=job_def,
        parameters={
            "localpath": local_path,
            "s3src": s3_source_prefix,
            "s3destination": dest_s3,
            "s3network_csv": s3_network_prefix_csv,
            "networktype": network_type,
            "threshold": str(positives_filter_threshold)

        },
        timeout={
            'attemptDurationSeconds': 86400 * 2
        }
    )

    print(response)


def submit_multiple(job_def, queue_name, s3_source_prefix, s3_destination_prefix, s3_network_prefix_csv, network_type,
                    positives_filter_threshold,
                    local_path="/data"):
    # Submit job for each prefix
    for s3_bucket, s3_key in list_files(s3_source_prefix):
        sleep(1)
        submit_job(job_def, queue_name, "s3://{}/{}".format(s3_bucket, s3_key), s3_destination_prefix,
                   s3_network_prefix_csv, network_type, positives_filter_threshold, local_path)


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
                        help="A csv list of s3 uri prefix that will contain the input/output data. e.g s3://mybucket/aws-batch-sample-python/")

    parser.add_argument("s3_dest_prefix",
                        help="The s3 uri prefix that will contain the input/output data. e.g s3://mybucket/aws-batch-sample-python/")

    parser.add_argument("s3_network_artifacts_prefix_csv",
                        help="The s3 uri prefix containing the network related artifacts. Makesure the name ends in / , e.g. s3://mybucket/model/")

    parser.add_argument("s3_network_type",
                        help="The type of network e.g. CnnPos, Cnn or Linear")

    parser.add_argument("--positives-filter-threshold",
                        help="Threshold to use", type=float, default=0.0)

    args = parser.parse_args()

    # Register job
    submit_multiple(args.job_name, args.queue, args.s3_source_prefix, args.s3_dest_prefix,
                    args.s3_network_artifacts_prefix_csv, args.s3_network_type, args.positives_filter_threshold)

    logger.info("Completed")
