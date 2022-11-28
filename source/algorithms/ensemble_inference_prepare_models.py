import argparse
import glob
import logging
import os
import sys
import tarfile
import uuid


class EnsembleInferencePrepareModels:
    """
    Sets up the directory structure
    """

    @property
    def logger(self):
        return logging.getLogger(__name__)

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

    def __call__(self, local_input_dir, local_output_dir):
        # Move all files to the working dir, extract tar.gz if required
        for full_file_path in glob.glob("{}/**/*.tar.gz".format(local_input_dir), recursive=True):
            cur_result_dir = os.path.join(local_output_dir, "{}".format(uuid.uuid4()))

            self.logger.info("Extracting file {} to {}".format(full_file_path, cur_result_dir))

            os.mkdir(cur_result_dir)

            self._extract_tar(full_file_path, cur_result_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
                        help="The input directory containing the models tar.gz",
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
    EnsembleInferencePrepareModels()(args.input_dir, args.dest_dir)

    logger.info("completed...")
