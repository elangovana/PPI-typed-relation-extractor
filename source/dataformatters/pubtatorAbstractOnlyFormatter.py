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
        # Writing uniue pubmeds only because Gnormplu requires unique ids
        uniqueids = set()
        for item in data_iter:
            pubmedid = label_func(item)
            if pubmedid in uniqueids: continue

            uniqueids.add(pubmedid)

            abstract = text_func(item).replace("\n", " ").replace("|", " ");

            # Remove unicode characters they cause issues with GNormPlus
            clean_abstract = abstract.encode('ascii', 'ignore').decode("utf-8")

            line = "{}|{}|{}\n\n".format(pubmedid, "a", clean_abstract)
            output_handle.write(line)
