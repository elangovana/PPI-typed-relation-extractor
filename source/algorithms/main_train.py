import argparse

import pandas as pd
import sys

from algorithms.RelationExtractionFactory import RelationExtractionFactory
import logging
from sklearn.model_selection import train_test_split
import numpy as np

def run(data_file, embedding_file, embed_dim, tmp_dir, epochs, interaction_type="phosphorylation"):
    logger = logging.getLogger(__name__)
    data = pd.read_json(data_file)
    class_size = 2

    logger.info("Running with interaction type {}".format(interaction_type))
    data = data.query('interactionType == "{}"'.format(interaction_type))

    labels = data[["isNegative"]]
    data = data[["pubmedabstract", "destAlias", "sourceAlias"]]

    data['destAlias'] = data['destAlias'].map(lambda x: ", ".join(x[0]))
    data['sourceAlias'] = data['sourceAlias'].map(lambda x: ", ".join(x[0]))


    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2,
                                                        random_state=777)

    logger.info("Training shape {}, test shape {}".format(X_train.shape, X_test.shape))

    # TODO: For the moment just filter one type of classification and do binary as true vs false
    with open(embedding_file, "r") as embedding:
        # Ignore the first line as it contains the number of words and vector dim
        head = embedding.readline()
        logger.info("The embedding header is {}".format(head))
        train_factory = RelationExtractionFactory(embedding_handle=embedding, embedding_dim=embed_dim, class_size=class_size,
                                                  output_dir=tmp_dir, ngram=1, epochs=epochs)
        train_factory(X_train, np.reshape(y_train.values.tolist(),(-1,)), X_test, np.reshape(y_test.values.tolist(),(-1,)))



if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("inputjson",
                        help="The input json data")
    parser.add_argument("embedding", help="The embeeing file")
    parser.add_argument("embeddim", help="the embed dim", type=int)

    parser.add_argument("outdir", help="The output dir")
    parser.add_argument("--epochs", help="The number of epochs", type=int)
    parser.add_argument("--interaction-type", help="The interction type", default="phosphorylation")

    args = parser.parse_args()

    run(args.inputjson,args.embedding,args.embeddim,
        args.outdir, args.epochs, args.interaction_type)




