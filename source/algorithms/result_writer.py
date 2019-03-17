"""
Writes results
"""
import datetime
import json
import logging
import os
import uuid


class ResultWriter:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, x, y_actual, y_pred, pos_label, output_dir, x_meta=None, filename_prefix="results"):
        from sklearn.metrics import confusion_matrix
        cnf_matrix = confusion_matrix(y_actual, y_pred)

        filename = os.path.join(output_dir,
                                "predictedvsactual_{}_{}.csv".format(str(uuid.uuid4()),
                                                                     datetime.datetime.strftime(datetime.datetime.now(),
                                                                                                format="%Y%m%d_%H%M%S")))

        if self.logger.isEnabledFor(logging.DEBUG):
            self._save_data(y_pred, y_actual, filename)

        self.logger.info("Confusion matrix, full output in {}: \n{}".format(filename, cnf_matrix))

    def dump_object(self, object, output_dir, filename_prefix):
        """
Dumps the object as a json to a file
        :param object:
        """
        filename = os.path.join(output_dir,
                                "{}_Objectdump_{}_{}.json".format(filename_prefix,
                                                                  datetime.datetime.strftime(datetime.datetime.now(),
                                                                                             format="%Y%m%d_%H%M%S"),
                                                                  str(uuid.uuid4())))

        with open(filename, "w") as o:
            json.dump(object, o)

    def _save_data(self, pred, actual, outfile):
        # Write to output
        with open(outfile, "w") as out:
            out.write("{},{}\n".format("actual", "pred"))
            for a, p in zip(actual, pred):
                out.write("{},{}\n".format(a, p))
