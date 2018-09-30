from typing import Callable, Any


class FastTextFormatter:

    def __call__(self, data_iter: iter, label_func: Callable[[Any], str], text_func: Callable[[Any], str],
                 output_handle):
        """
Formats the input list into the format expected by fast text which is label followed by the text
__label__mylabel RawText
        :param output_handle: A file like object to which the formatted output is written into
        :param text_func: A function  which accepts an item from the collection of records and returns the text value
        :param label_func: A function which accepts an item from the collection of records and returns the label value
        :param data_iter: A iterable collection of records
        """
        for item in data_iter:
            label = "__label__{}".format(label_func(item))
            text = text_func(item).replace("\n", " ")
            line = "{} {}\n".format(label, text)
            output_handle.write(line)
