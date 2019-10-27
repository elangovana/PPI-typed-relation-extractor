import argparse
import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from algorithms.dataset_factory import DatasetFactory
from algorithms.main_train import run
from algorithms.network_factory_locator import NetworkFactoryLocator


def k_fold_unique_pubmed(data_file, n_splits=10):
    kf = KFold(n_splits=n_splits, random_state=777, shuffle=True)
    df = pd.read_json(data_file)
    unique_docids = df.docid.unique()
    stratified = [df.query("docid == '{}'".format(p))['isValid'].iloc[0] for p in unique_docids]
    for train_index, test_index in kf.split(unique_docids, groups=stratified):
        train_doc, test_doc = unique_docids[train_index], unique_docids[test_index]
        train = df[df['docid'].isin(train_doc)]
        val = df[df['docid'].isin(test_doc)]

        yield (train, val)


def k_fold(data_file, n_splits=10):
    kf = KFold(n_splits=n_splits, random_state=777, shuffle=True)
    df = pd.read_json(data_file)

    for train_index, test_index in kf.split(df, groups=df["isValid"]):
        train, val = df.iloc[train_index], df.iloc[test_index]

        yield (train, val)


def run_k_fold(dataset_factory_name, network_factory_name, train_file, embedding_file, embed_dim, model_dir, out_dir,
               epochs, earlystoppingpatience, additionalargs):
    logger = logging.getLogger(__name__)


    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(out_dir))

    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(model_dir))

    k_val_results = []

    for k, (train_df, val_df) in enumerate(k_fold(train_file)):
        with tempfile.NamedTemporaryFile() as tmp_train_spilt_file:
            train_df.to_json(tmp_train_spilt_file.name)
            with tempfile.NamedTemporaryFile() as tmp_val_split_file:
                val_df.to_json(tmp_val_split_file.name)

                logger.info("Running fold {}".format(k))

                val_results, val_actuals, val_predicted = run(dataset_factory_name, network_factory_name,
                                                              tmp_train_spilt_file.name, tmp_val_split_file.name,
                                                              embedding_file, embed_dim, model_dir, out_dir,
                                                              epochs,
                                                              earlystoppingpatience, additionalargs)

            k_val_results.append(val_results)

            logger.info("Fold {}, F-score is {}".format(k, val_results))

    print("Average F-score", np.asarray(k_val_results).mean())


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="The dataset type", choices=DatasetFactory().dataset_factory_names,
                        required=True)

    parser.add_argument("--network", help="The network type", choices=NetworkFactoryLocator().factory_names,
                        default=NetworkFactoryLocator().factory_names[0])

    parser.add_argument("--trainfile",
                        help="The input train file wrt to train  dir", required=True)

    parser.add_argument("--traindir",
                        help="The input train  dir", default=os.environ.get("SM_CHANNEL_TRAIN", "."))

    parser.add_argument("--embeddingfile", help="The embedding file wrt to the embedding dir", required=True)

    parser.add_argument("--embeddingdir", help="The embedding dir", default=os.environ.get("SM_CHANNEL_EMBEDDING", "."))

    parser.add_argument("--outdir", help="The output dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "."))

    parser.add_argument("--modeldir", help="The output dir", default=os.environ.get("SM_MODEL_DIR", "."))

    parser.add_argument("--embeddim", help="the embed dim", type=int, required=True)

    parser.add_argument("--epochs", help="The number of epochs", type=int, default=10)

    parser.add_argument("--earlystoppingpatience", help="The number of pateince epochs epochs", type=int, default=10)

    parser.add_argument("--interaction-type", help="The interction type", default=None)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args, additional = parser.parse_known_args()

    # Convert additional args into dict
    print(additional)
    additional_dict = {}
    for i in range(0, len(additional), 2):
        additional_dict[additional[i].lstrip("--")] = additional[i + 1]

    print(args.__dict__)
    print(additional_dict)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    trainjson = os.path.join(args.traindir, args.trainfile)
    embeddingfile = os.path.join(args.embeddingdir, args.embeddingfile)
    run_k_fold(args.dataset, args.network, trainjson, embeddingfile, args.embeddim,
               args.modeldir, args.outdir, args.epochs, args.earlystoppingpatience, additional_dict)
