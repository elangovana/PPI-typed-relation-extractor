import logging

import pandas as pd
from sklearn.model_selection import StratifiedKFold


#
# def generate_split_by_doc(df, label_field_name, docid_field_name, split):
#     random.seed(777)
#     unique_doc = df[docid_field_name].unique()
#     unique_labels = df[label_field_name].unique()
#     random.shuffle(unique_doc)
#
#     result = {}
#
#     label_counts_raw = df[label_field_name].value_counts()
#     label_counts_percentage =  label_counts_raw * 100 /sum(label_counts_raw.values)
#
#
#     split_size= len(unique_labels)//split
#
#    # print(apprixmiate_size_per_label)
#
#     reuslt = df.groupby([label_field_name,docid_field_name]).size()
#  #  print(reuslt)
#
#     for d in unique_doc:
#         if d not in result:
#             result[d] = {l: 0 for l in unique_labels}
#
#         label_value_counts = df.query("{} == '{}'".format(docid_field_name, d))[label_field_name].value_counts()
#         #print(label_value_counts)

def label_distribution(df, label_field_name):
    label_counts_raw = df[label_field_name].value_counts()
    label_counts_percentage = label_counts_raw * 100 / sum(label_counts_raw.values)

    return label_counts_percentage


def k_fold_unique_doc(data_file, label_field_name, docid_field_name, n_splits=10):
    logger = logging.getLogger(__name__)
    logger.info("Splitting such that the {} is unique across datasets".format(docid_field_name))
    kf = StratifiedKFold(n_splits=n_splits, random_state=777, shuffle=True)
    df = pd.read_json(data_file)
    unique_docids = df.docid.unique()
    # Do a approx so that the labels are somewhat stratified in the split
    approx_y = [df.query("{} == '{}'".format(docid_field_name, p))[label_field_name].iloc[0] for p in unique_docids]
    for train_index, test_index in kf.split(unique_docids, approx_y):
        train_doc, test_doc = unique_docids[train_index], unique_docids[test_index]
        train = df[df[docid_field_name].isin(train_doc)]
        val = df[df[docid_field_name].isin(test_doc)]

        logger.info("Train split label distribution {} ".format(
            str(label_distribution(train, label_field_name)).replace("\n", "\t")))
        logger.info("Validation split label distribution {} ".format(
            str(label_distribution(val, label_field_name)).replace("\n", "\t")))
        yield (train, val)


def k_fold_ignore_doc(data_file, label_field_name, n_splits=10):
    kf = StratifiedKFold(n_splits=n_splits, random_state=777, shuffle=True)
    df = pd.read_json(data_file)
    logger = logging.getLogger(__name__)

    for train_index, test_index in kf.split(df, df[label_field_name]):
        train, val = df.iloc[train_index], df.iloc[test_index]

        logger.info("Train split label distribution {} ".format(
            str(label_distribution(train, label_field_name)).replace("\n", "\t")))
        logger.info("Validation split label distribution {} ".format(
            str(label_distribution(val, label_field_name)).replace("\n", "\t")))

        yield (train, val)


def k_fold(data_file, label_field_name, docid_field_name=None, n_splits=10):
    if docid_field_name is None:
        yield from k_fold_ignore_doc(data_file, label_field_name, n_splits)
    else:
        yield from k_fold_unique_doc(data_file, label_field_name, docid_field_name, n_splits)
