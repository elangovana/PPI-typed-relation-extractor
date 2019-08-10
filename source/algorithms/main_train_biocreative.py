import logging

import numpy as np
import pandas as pd


def prepare_data(self_relations_filter, data_df):
    logger = logging.getLogger(__name__)

    if self_relations_filter:
        logger.info("Removing self relations")

        data_df = data_df.query('participant1 != participant2')
    labels = data_df[["isValid"]]
    data_df = data_df[["abstract", "participant1", "participant2"]]

    labels = np.reshape(labels.values.tolist(), (-1,))
    return data_df, labels


def up_sample_minority(train_df, self_relations_filter):
    logger = logging.getLogger(__name__)

    # True is the minority class
    if self_relations_filter:
        logger.info("Removing self relations")
        train_df = train_df.query('participant1 != participant2')

    train_dat_0s = train_df.query('isValid == False')
    train_dat_1s = train_df.query('isValid == True')

    rep_1 = [train_dat_1s for x in range((train_dat_0s.shape[0] // train_dat_1s.shape[0]) // 3)]

    keep_1s = pd.concat(rep_1, axis=0)

    train_dat = pd.concat([keep_1s, train_dat_0s], axis=0)
    return train_dat
