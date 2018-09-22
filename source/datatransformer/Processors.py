class Processors:
    def __init__(self, processor_list):
        self.processor_list = processor_list

    def process(self, imex_file_name, doc_index, doc):
        for processor in self.processor_list:
            processor.process(imex_file_name, doc_index, doc)