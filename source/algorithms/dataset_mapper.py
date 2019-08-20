import sys


def str_to_dataset_class(class_name):
    valid_datasets_module = get_datasets()
    if class_name not in valid_datasets_module:
        raise "Unknown dataset, expected one of these {} ".format(" , ".join(valid_datasets_module.keys()))

    module_name = valid_datasets_module[class_name]
    return getattr(sys.modules[module_name], class_name)


def get_datasets():
    valid_datasets_module = {"PPIDataset": "algorithms.PpiDataset", "PpiAimedDataset": "algorithms.PpiAimedDataset"}
    return valid_datasets_module
