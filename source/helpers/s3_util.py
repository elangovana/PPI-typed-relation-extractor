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
import datetime
import glob
import logging
import os
from multiprocessing.dummy import Pool as ThreadPool

import boto3

from helpers.external_file_base import ExternalFileBase


class S3Util(ExternalFileBase):

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def uploadfile(self, localpath, remote_path):
        """
    Uploads a file to s3
        :param localpath: The local path
        :param remote_path: The s3 path in format s3://mybucket/mydir/mysample.txt
        """

        bucket, key = self._get_bucketname_key(remote_path)

        if key.endswith("/"):
            key = "{}{}".format(key, os.path.basename(localpath))

        self._logger.info("Uploading file {} s3://{}/{}".format(localpath, bucket, key))

        s3 = boto3.client('s3')
        s3.upload_file(localpath, bucket, key)

    def _get_bucketname_key(self, uripath):
        assert uripath.startswith("s3://")

        path_without_scheme = uripath[5:]
        bucket_end_index = path_without_scheme.find("/")

        bucket_name = path_without_scheme
        key = "/"
        if bucket_end_index > -1:
            bucket_name = path_without_scheme[0:bucket_end_index]
            key = path_without_scheme[bucket_end_index + 1:]

        return bucket_name, key

    def download_file(self, remote_path, local_dir):
        """
        Download a single file from s3
        :param remote_path: The remote s3 file
        :param local_dir: The local directory to save the file to
        :return:
        """
        bucket, key = self._get_bucketname_key(remote_path)

        s3 = boto3.client('s3')

        local_file = os.path.join(local_dir, remote_path.split("/")[-1])

        s3.download_file(bucket, key, local_file)

    def download_object(self, remote_path):
        """
        Downloads binary bytes from s3 without saving file
        :param remote_path: The remote s3 path
        :return: returns binary bytes from s3 without saving file
        """
        bucket, key = self._get_bucketname_key(remote_path)

        s3 = boto3.client('s3')

        s3_response_object = s3.get_object(Bucket=bucket, Key=key)
        object_content = s3_response_object['Body'].read()

        return object_content

    def list_files(self, remote_path):
        """
Lists the files in s3
        :param remote_path: The s3 uri, e.g. s3://mybucket/prefix/
        :return: List of files
        """
        assert remote_path.startswith("s3://")
        assert remote_path.endswith("/")

        bucket, key = self._get_bucketname_key(remote_path)

        s3 = boto3.resource('s3')

        bucket = s3.Bucket(name=bucket)

        return ((o.bucket_name, o.key) for o in bucket.objects.filter(Prefix=key))

    def upload_files(self, local_dir, remote_path, num_threads=20):
        """
Uploads the files in local directory to s3
        :param local_dir: The local directory
        :param remote_path: The remote s3 prefix
        :param num_threads: The number of parallel threads used to upload to s3
        """
        input_tuples = ((f, remote_path) for f in glob.glob("{}/*".format(local_dir)))

        with ThreadPool(num_threads) as pool:
            pool.starmap(self.uploadfile, input_tuples)

    def download_files(self, remote_path, local_dir, num_threads=20):
        """
    Downloads the files from s3 to  local directory
        :param remote_path: The remote s3 path prefix
        :param local_dir: The local directory
        :param num_threads: The number of parallel downloads
        :return: 
        """
        input_tuples = (("s3://{}/{}".format(s3_bucket, s3_key), local_dir) for s3_bucket, s3_key in
                        self.list_files(remote_path))

        with ThreadPool(num_threads) as pool:
            pool.starmap(self.download_file, input_tuples)

    def download_objects(self, s3_prefix, num_threads=20):
        s3_files = ("s3://{}/{}".format(s3_bucket, s3_key) for s3_bucket, s3_key in self.list_files(s3_prefix))

        with ThreadPool(num_threads) as pool:
            results = pool.map(self.download_object, s3_files)

        return results

    def _get_directory_size(self, start_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("s3url",
                        help="The s3 path. to download from e.g. s3://mybuck/prefix/")
    parser.add_argument("localdir",
                        help="The local directory to save the file to")

    args = parser.parse_args()

    print("Starting download...into memory without saving to filefile")
    start = datetime.datetime.now()

    s3_util = S3Util()
    s3_util.download_files(args.s3url, args.localdir, num_threads=15)
    end = datetime.datetime.now()

    download_time = end - start

    print("Total time in seconds to download  {} seconds".format(download_time.total_seconds()))
