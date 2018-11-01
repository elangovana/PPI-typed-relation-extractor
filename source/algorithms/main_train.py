import argparse

import pandas as pd

from algorithms.RelationExtractionFactory import RelationExtractionFactory
import logging
from sklearn.model_selection import train_test_split


def run(data_file, embedding_file, tmp_dir, interaction_type="phosphorylation"):
    logger = logging.getLogger(__name__)
    data = pd.read_json(data_file)
    class_size = 2

    logger.info("Running with interaction type {}".format(interaction_type))
    data = data.query('interactionType == "{}"'.format(interaction_type))

    labels = data[["isNegative"]]
    data = data[["interactionType", "pubmedabstract", "destAlias", "sourceAlias"]]

    X_train, X_test, X_train, y_test = train_test_split(data, labels, train_size=.8,
                                                        random_state=777)

    # TODO: For the moment just filter one type of classification and do binary as true vs false
    with open(embedding_file, "r") as embedding:
        train_factory = RelationExtractionFactory(embedding, embedding_dim=300, class_size=class_size,
                                                  output_dir=tmp_dir, ngram=1)
        train_factory(X_train, X_train, X_test, y_test)



if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputjson",
                        help="The input json data")
    parser.add_argument("embedding", help="The embeeing file")
    parser.add_argument("out-dir", help="The output dir")
    parser.add_argument("--interaction-type", help="The interction type", default="phosphorylation")

    args = parser.parse_args()

    run(args.inputjson,args.embedding,
        args.out_dir, args.interaction_type)



