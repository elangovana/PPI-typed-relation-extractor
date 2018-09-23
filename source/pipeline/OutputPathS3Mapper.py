import logging
import os
import tempfile

import boto3


class OutputPathS3Mapper:

    @staticmethod
    def get_scheme():
        return "s3"

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, uripath, localpath):
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
        # TODO check local path is dir
        if key.endswith("/"):
            pass
             # TODO write code local path is dir
        else:
            local_path = self.upload_single_file(bucket_name, key, localpath)
            return local_path



    def upload_single_file(self, bucket_name, key, tempath):
        client = boto3.resource('s3')
        local_path = tempath
        self.logger.info("Uploading to bucket {}, {} to file {}".format(bucket_name, key, local_path))
        with open(tempath, "rb") as data:
            client.Bucket(bucket_name).put_object(key, data)

