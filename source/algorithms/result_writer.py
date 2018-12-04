"""
Writes results
"""
import datetime
import logging
import os


class ResultWriter:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, x, y_actual, y_pred, pos_label, output_dir, x_meta=None, filename_prefix="results"):
        from sklearn.metrics import confusion_matrix
        cnf_matrix = confusion_matrix(y_actual, y_pred)

        filename = os.path.join(output_dir,
                                "predictedvsactual_{}".format(
                                    datetime.datetime.strftime(datetime.datetime.now(), format="%Y%m%d_%H%M%S")))
        self.save_data(y_pred, y_actual, filename)
        self.logger.info("Confusion matrix, full output in {}: \n{}".format(filename, cnf_matrix))

    def save_data(self, pred, actual, outfile):
        # Write to output

        with open(outfile, "w") as out:
            for a, p in zip(actual, pred):
                out.write("{},{}\n".format(a, p))
