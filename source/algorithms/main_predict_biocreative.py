import logging
import os

import pandas as pd


def prepare_data(self_relations_filter, data_df):
    if self_relations_filter:
        data_df = data_df.query('participant1 != participant2')
    data_df = data_df[["abstract", "participant1", "participant2"]]

    return data_df


def run(network, data_file, artifactsdir, out_dir, self_relations_filter=True):
    logger = logging.getLogger(__name__)

    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(out_dir))

    logger.info("Running with self relations filter {}, network {}".format(self_relations_filter, network))

    df = pd.read_json(data_file, dtype={'docid': 'str'})
    logger.info("Data size after load: {}".format(df.shape))

    df_prep = prepare_data(self_relations_filter, df)
    logger.info("Data size after prep: {}".format(df_prep.shape))
