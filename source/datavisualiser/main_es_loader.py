import argparse
import logging
import os

from aws_requests_auth.aws_auth import AWSRequestsAuth

from datavisualiser.ImexJsonProcessorElasticSearchLoader import ImexJsonProcessorElasticSearchLoader
from dataloader.ImexJsonProcessorFileWriter import ImexJsonProcessorFileWriter
from datavisualiser.elasticSearchWrapper import connectES

from dataloader.Processors import Processors
from dataloader.imexDataPreprocessor import ImexDataPreprocessor


def bulk_run(data_dir, processor):
    logger = logging.getLogger(__name__)
    # Get xml files in dir
    for imex_file_name in os.listdir(data_dir):
        if not imex_file_name.endswith(".xml"):
            continue

        # Assuming all xml files are valid imex files.
        full_path = os.path.join(data_dir, imex_file_name)
        logger.info("Processing file {}".format(full_path))
        data_processor = ImexDataPreprocessor()

        with open(full_path, "rb") as xmlhandle:
            i = 0
            for doc in data_processor.run_pipeline(xmlhandle):
                i = i + 1
                processor.process(imex_file_name, i, doc)


def run(input_dir, elastic_search_domain, aws_region, aws_access_key_id, aws_secret_access_key, out_dir):
    auth = AWSRequestsAuth(aws_access_key=aws_access_key_id,
                           aws_secret_access_key=aws_secret_access_key,
                           aws_token="",
                           aws_host=elastic_search_domain,
                           aws_region=aws_region,
                           aws_service='es')
    esclient = connectES(elastic_search_domain, auth)

    ##Consolidate all processors
    filewriter_processor = ImexJsonProcessorFileWriter(out_dir)
    es_processor = ImexJsonProcessorElasticSearchLoader(esclient)
    processors = Processors([filewriter_processor, es_processor])

    # Run
    bulk_run(input_dir, processors)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help="The input directory containing the imex files")
    parser.add_argument("out_dir", help="The output dir")
    parser.add_argument("--elasticsearchdomain",
                        help="Elastic Search Domain, e.g search-amy-pwlposkclcmld3azaydylcbugy.us-east-2.es.amazonaws.com",
                        default=os.environ.get("elasticsearch_domain_name", None))
    parser.add_argument("--awsregion", help="AWS Region , e.g us-east-1", default=os.environ.get("AWS_REGION", None))
    parser.add_argument("--awsaccesskeyid", help="AWS Region , e.g us-east-1",
                        default=os.environ.get("AWS_ACCESS_KEY_ID", None))
    parser.add_argument("--awssecretaccesskey", help="AWS Region , e.g us-east-1",
                        default=os.environ.get("AWS_SECRET_ACCESS_KEY", None))

    args = parser.parse_args()

    run(args.input_dir, args.elasticsearchdomain, args.awsregion, args.awsaccesskeyid, args.awssecretaccesskey,
        args.out_dir)
