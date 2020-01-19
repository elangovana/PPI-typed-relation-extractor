class ExternalFileBase:

    def uploadfile(self, localpath, remote_path):
        """
    Uploads a file to remote dest
        :param localpath: The local path
        :param remote_path: The remote path, e.g. aws s3 s3://mybucket/mydir/mysample.txt
        """
        raise NotImplementedError()

    def download_file(self, remote_path, local_dir):
        raise NotImplementedError()

    def download_object(self, remote_path):
        raise NotImplementedError()

    def list_files(self, remote_path):
        raise NotImplementedError()

    def upload_files(self, local_dir, remote_path, num_threads=20):
        raise NotImplementedError()

    def download_files(self, remote_path, local_dir, num_threads=20):
        raise NotImplementedError()

    def download_objects(self, remote_path, num_threads=20):
        raise NotImplementedError()
