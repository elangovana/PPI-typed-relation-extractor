import argparse
import logging

from gensim.models.keyedvectors import KeyedVectors


def convert(bin_path, output_text_path):
    model = KeyedVectors.load_word2vec_format(bin_path, binary=True)
    model.save_word2vec_format(output_text_path, binary=False)


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("inputwordtovecbin",
                        help="The input word to vec binary formatted file")
    parser.add_argument("outfile", help="The output file name")
    args = parser.parse_args()

    convert(args.inputwordtovecbin, args.outfile)
