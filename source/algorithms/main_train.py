
import argparse
import logging
import os
import sys

from algorithms.TrainWorkflow import TrainWorkflow
from algorithms.dataset_factory import DatasetFactory
from algorithms.network_factory_locator import NetworkFactoryLocator


def run(dataset_factory_name, network_factory_name, train_file, val_file, test_file, embedding_file, embed_dim,
        model_dir, out_dir,
        epochs,
        earlystoppingpatience, additionalargs):
    logger = logging.getLogger(__name__)

    if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(out_dir))

    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        raise FileNotFoundError("The path {} should exist and must be a directory".format(model_dir))

    workflow = TrainWorkflow(dataset_factory_name=dataset_factory_name, network_factory_name=network_factory_name,
                             embedding_dim=embed_dim, embedding_file=embedding_file,
                             model_dir=model_dir, out_dir=out_dir, epochs=epochs,
                             patience_epochs=earlystoppingpatience,
                             extra_args=additionalargs)

    val_results, val_actuals, val_predicted = workflow(train_file, val_file, test_file=test_file)



    return val_results, val_actuals, val_predicted



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

    parser.add_argument("--valfile",
                        help="The input val file wrt to val  dir", required=True)

    parser.add_argument("--valdir",
                        help="The input val dir", default=os.environ.get("SM_CHANNEL_VAL", "."))

    parser.add_argument("--testfile",
                        help="The input test file wrt to val  dir", required=False, default=None)

    parser.add_argument("--testdir",
                        help="The input test dir", default=os.environ.get("SM_CHANNEL_TEST", "."))

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
    valjson = os.path.join(args.valdir, args.valfile)

    testjson = None
    if args.testfile is not None:
        testjson = os.path.join(args.testdir, args.testfile)

    embeddingfile = os.path.join(args.embeddingdir, args.embeddingfile)
    run(args.dataset, args.network, trainjson, valjson, testjson, embeddingfile, args.embeddim,
        args.modeldir, args.outdir, args.epochs, args.earlystoppingpatience, additional_dict)
