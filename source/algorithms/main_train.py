import argparse

import pandas as pd

from algorithms.RelationExtractionFactory import RelationExtractionFactory
import logging
from sklearn.model_selection import train_test_split


def run(data_file, embedding_file, embed_dim, tmp_dir, interaction_type="phosphorylation"):
    logger = logging.getLogger(__name__)
    data = pd.read_json(data_file)
    class_size = 2

    logger.info("Running with interaction type {}".format(interaction_type))
    data = data.query('interactionType == "{}"'.format(interaction_type))

    labels = data[["isNegative"]]
    data = data[["pubmedabstract", "destAlias", "sourceAlias"]]

    data['destAlias'] = data['destAlias'].map(lambda x: ", ".join(x[0]))
    data['sourceAlias'] = data['sourceAlias'].map(lambda x: ", ".join(x[0]))


    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=.8,
                                                        random_state=777)

    # TODO: For the moment just filter one type of classification and do binary as true vs false
    with open(embedding_file, "r") as embedding:
        train_factory = RelationExtractionFactory(embedding, embedding_dim=embed_dim, class_size=class_size,
                                                  output_dir=tmp_dir, ngram=1)
        train_factory(X_train, y_train, X_test, y_test)



if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputjson",
                        help="The input json data")
    parser.add_argument("embedding", help="The embeeing file")
    parser.add_argument("embeddim", help="the embed dim", type=int)

    parser.add_argument("outdir", help="The output dir")
    parser.add_argument("--interaction-type", help="The interction type", default="phosphorylation")

    args = parser.parse_args()

    run(args.inputjson,args.embedding,args.embeddim,
        args.outdir, args.interaction_type)




