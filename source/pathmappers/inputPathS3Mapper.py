import logging
import os
import tempfile

import boto3


class InputPathS3Mapper:

    @staticmethod
    def get_scheme():
        return "s3"

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, uripath):
        # check if path uri
        self.logger.info("Downloding from Uri {}".format(uripath))
        index = uripath.find("://")

        ## local file
        if index == -1: return uripath

        # Check if s3
        scheme = uripath[0:index]
        if scheme.lower() != "s3":
            return ""

        # get bucket and key
        path_without_scheme = uripath[index + 3:]

        bucket_end_index = path_without_scheme.find("/")
        bucket_name = path_without_scheme
        key = "/"
        if bucket_end_index > -1:
            bucket_name = path_without_scheme[0:bucket_end_index]
            key = path_without_scheme[bucket_end_index + 1:]

        # download file from s3 to temp directory and return path
        tempath = tempfile.mkdtemp("data")
        if key.endswith("/"):
            self.download_directory(bucket_name, key, tempath)
            return tempath
        else:
            parts = path_without_scheme.split("/")
            file_name = parts[len(parts) - 1]
            local_path = os.path.join(tempath, file_name)

            local_path = self.download_single_file(bucket_name, key, tempath)
            return local_path

    def download_directory(self, bucket_name, key, tempath):
        # this is a directory, process using manifest file
        assert key.endswith("/")

        # download manifest
        manifest_key = key + "manifest.txt"
        local_path = os.path.join(tempath, "manifest.txt")

        self.download_single_file(bucket_name, manifest_key, local_path)
        with open(local_path, "r") as m:
            files = m.readlines()

        # Download each file, assumes relative path
        for f in files:
            # remove leading / or ./
            if f.startswith("/"):
                f = f[1:]
            elif f.startswith("./"):
                f = f[2:]
            f = f.strip("\n")
            file_key = key + f

            local_path = os.path.join(tempath, f)
            self.download_single_file(bucket_name, file_key, local_path)

    def download_single_file(self, bucket_name, key, tempath):
        client = boto3.resource('s3')
        local_path = tempath
        self.logger.info("Downloading from bucket {}, {} to file {}".format(bucket_name, key, local_path))

        client.Bucket(bucket_name).download_file(key, local_path)
        return local_path
