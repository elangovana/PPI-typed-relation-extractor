import argparse
import logging

import sys


class CollobertEmbeddingFormatter:
    """
    This is to convert embeddings created in https://ronan.collobert.com/pub/matos/2014_hellinger_eacl.pdf .
    Formats the vocab, embedding files/handles such that the resulting file is of the format
    word1 emdedding1..
    word2 emdedding2..


    Useful links
    =============
     * http://www.lebret.ch/words/embeddings/ Actual embeddings
    """

    def __init__(self, vocab_handle_or_file, embedding_handler_or_file):
        self.embedding_handler_or_file = embedding_handler_or_file
        self.vocab_handle_or_file = vocab_handle_or_file

    def convert(self, destination_handle):
        embed_handle = self.embedding_handler_or_file
        vocab_handle = self.vocab_handle_or_file
        if isinstance(self.embedding_handler_or_file, str):
            with  open(self.embedding_handler_or_file, "r") as e:
                embed_handle = e

                if isinstance(self.vocab_handle_or_file, str):
                    with  open(self.vocab_handle_or_file, "r") as v:
                        vocab_handle = v
                        self._convert(vocab_handle, embed_handle, destination_handle)
                else:
                    self._convert(vocab_handle, embed_handle, destination_handle)
        else:
            self._convert(vocab_handle, embed_handle, destination_handle)

    def convert_to_file(self, destination_file):
        with open(destination_file, "w") as d:
            self.convert(d)

    def _convert(self, vocab_handle, embed_handle, destination_handle):
        word_count = 0
        embedding_size = -1
        destination_handle.write("{:010d} {:010d}\n".format(0, 0))
        for w, e in zip(vocab_handle, embed_handle):
            destination_handle.write("{} {}".format(w.strip("\n"), e))
            word_count += 1
            if embedding_size == -1:
                embedding_size = len(e.split(" "))

        destination_handle.seek(0)
        destination_handle.write("{:010d} {:010d}\n".format(word_count, embedding_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocabfile",
                        help="The vocab file", required=True)

    parser.add_argument("--embedfile",
                        help="The embed file", required=True)

    parser.add_argument("--outputfile",
                        help="The outputfile to write to", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Convert additional args into dic
    print(args.__dict__)
    print("Starting..")

    formatter = CollobertEmbeddingFormatter(vocab_handle_or_file=args.vocabfile,
                                            embedding_handler_or_file=args.embedfile)
    formatter.convert_to_file(args.outputfile)

    print("Completed successfully")
