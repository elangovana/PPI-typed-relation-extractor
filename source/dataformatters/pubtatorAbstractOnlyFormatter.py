from typing import Callable, Any


class PubtatorAbstractOnlyFormatter:

    def __call__(self, data_iter: iter, label_func: Callable[[Any], str], text_func: Callable[[Any], str],
                 output_handle):
        """
Formats the data in pubtator abstract only format.
        :param output_handle: A file like object to which the formatted output is written into
        :param text_func: A function  which accepts an item from the collection of records and returns the abstract value
        :param label_func: A function which accepts an item from the collection of records and returns the pubmedid value
        :param data_iter: A iterable collection of records
        """
        uniqueids = set()
        for item in data_iter:
            pubmedid = label_func(item)
            if pubmedid in uniqueids: continue

            uniqueids.add(pubmedid)
            abstract = text_func(item).replace("\n", " ").replace("|", " ")

            line = "{}|{}|{}\n\n".format(pubmedid, "a", abstract)
            output_handle.write(line)
