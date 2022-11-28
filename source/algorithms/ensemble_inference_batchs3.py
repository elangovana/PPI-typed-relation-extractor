# *****************************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************
import argparse
import logging
import os
import shutil
import sys
import tarfile
import tempfile

from helpers.s3_util import S3Util


class EnsembleInferenceBatchS3:
    """
    Sets up the directory structure
    """

    def __init__(self, external_file_source=None):
        self._external_file_source = external_file_source or S3Util()

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def external_file_source(self):
        return self._external_file_source

    @external_file_source.setter
    def external_file_source(self, value):
        self._external_file_source = value

    def _extract_tar(self, tar_gz_file, dest_dir):
        with  tarfile.open(tar_gz_file) as tf:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tf, dest_dir)

    def __call__(self, remote_model_path_list, local_dir=None):

        # Set up working dir
        result_dir = local_dir

        with tempfile.TemporaryDirectory() as tmpdir:

            # Download files
            for i, remote_path in enumerate(remote_model_path_list):

                cur_result_dir = os.path.join(result_dir, "{}".format(i))
                cur_tmp_dir = os.path.join(tmpdir, "tmp_{}".format(i))

                os.mkdir(cur_result_dir)
                os.mkdir(cur_tmp_dir)

                self.logger.info("Downloading {} to {}".format(remote_path, cur_tmp_dir))

                self.external_file_source.download_file(remote_path, cur_tmp_dir)

                self.logger.info("Moving results  {} to {}".format(remote_path, cur_result_dir))

                # Move all files to the working dir, extract tar.gz if required
                for f in os.listdir(cur_tmp_dir):

                    full_file_path = os.path.join(cur_tmp_dir, f)
                    if full_file_path.endswith("tar.gz"):
                        self._extract_tar(full_file_path, cur_result_dir)
                    else:
                        dest_path = os.path.join(cur_result_dir, os.path.basename(f))

                        shutil.move(full_file_path, dest_path)

            return result_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-comma-sep-path",
                        help="A comma separated list of s3 paths. E.g. s3://bucket/folder_a,s3://bucket/folder_b",
                        required=True)

    parser.add_argument("--dest-dir",
                        help="The destination dir to save to",
                        required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)

    logger.info("Starting the job...")
    s3_input_list = [p.strip() for p in args.input_comma_sep_path.split(",")]
    EnsembleInferenceBatchS3()(s3_input_list, args.dest_dir)

    logger.info("completed...")
