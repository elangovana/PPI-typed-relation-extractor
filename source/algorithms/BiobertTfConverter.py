import argparse
import os

from pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch


class BioBertConverter:

    def __call__(self, base_dir, output_dir):
        convert_tf_checkpoint_to_pytorch(
            os.path.join(base_dir, 'model.ckpt-1000000'),
            os.path.join(base_dir, 'bert_config.json'),
            os.path.join(output_dir, 'pytorch_bio_bert_model.bin'))


if "__main__" == __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument("--modeldir",
                        help="The model dir that contains the pretrained biobert", required=True)

    parser.add_argument("--outputdir", help="The output dir", required=True)

    args = parser.parse_args()

    BioBertConverter()(args.modeldir, args.outputdir)

    print("Completed... model output written to {}".format(args.outputdir))
