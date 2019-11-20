import logging

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def k_fold_unique_doc(data_file, docid_field_name, n_splits=10):
    logger = logging.getLogger(__name__)
    logger.info("Splitting such that the {} is unique across datasets".format(docid_field_name))
    kf = KFold(n_splits=n_splits, random_state=777, shuffle=True)
    df = pd.read_json(data_file)
    unique_docids = df.docid.unique()
    for train_index, test_index in kf.split(unique_docids):
        train_doc, test_doc = unique_docids[train_index], unique_docids[test_index]
        train = df[df[docid_field_name].isin(train_doc)]
        val = df[df[docid_field_name].isin(test_doc)]

        yield (train, val)


def k_fold_ignore_doc(data_file, label_field_name, n_splits=10):
    kf = StratifiedKFold(n_splits=n_splits, random_state=777, shuffle=True)
    df = pd.read_json(data_file)

    for train_index, test_index in kf.split(df, df[label_field_name]):
        train, val = df.iloc[train_index], df.iloc[test_index]

        yield (train, val)


def k_fold(data_file, label_field_name, docid_field_name=None, n_splits=10):
    if docid_field_name is None:
        yield from k_fold_ignore_doc(data_file, label_field_name, n_splits)
    else:
        yield from k_fold_unique_doc(data_file, docid_field_name, n_splits)
