import argparse
import os
from shutil import copyfile

from pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch


class BioBertConverter:
    """
    Wrapper around pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch  to convert Pytorch pretrained bert to pytorch
    """

    def __call__(self, base_dir, output_dir):
        config_file = os.path.join(base_dir, 'bert_config.json')
        config_file_dest = os.path.join(output_dir, 'bert_config.json')
        # Convert
        convert_tf_checkpoint_to_pytorch(
            os.path.join(base_dir, 'model.ckpt-1000000'),
            config_file,
            os.path.join(output_dir, 'pytorch_model.bin'))
        # copy config file
        copyfile(config_file, config_file_dest)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument("--modeldir",
                        help="The model dir that contains the pretrained biobert", required=True)

    parser.add_argument("--outputdir", help="The output dir", required=True)

    args = parser.parse_args()

    BioBertConverter()(args.modeldir, args.outputdir)

    print("Completed... model output written to {}".format(args.outputdir))
