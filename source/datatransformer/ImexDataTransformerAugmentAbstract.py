from dataextractors.PubmedAbstractExtractor import PubmedAbstractExtractor


class ImexDataTransformerAugmentAbstract:
    def __init__(self):
        self.pubmed_extractor = None

    @property
    def pubmed_extractor(self):
        self.__pubmed_extractor__ = self.__pubmed_extractor__ or PubmedAbstractExtractor()
        return self.__pubmed_extractor__

    @pubmed_extractor.setter
    def pubmed_extractor(self, value):
        self.__pubmed_extractor__ = value

    def transform(self, input_iterable: iter, id_key: str = "pubmedid",
                  abstract_key: str = "pubmedabstract") -> iter:
        """
Adds the pubmed abstract to each item based on the pubmed id
        :param input_iterable: A iterable list of dictionary
        :param id_key: The key in the input identifying the pubmedid used to extract the abstract
        :param abstract_key: The key to use for the resulting abstract
        """
        for item in input_iterable:
            pubmed_details_list = self.pubmed_extractor.extract_abstract_by_pubmedid([item[id_key]])
            # expect only one record in the array
            assert len(pubmed_details_list) == 1
            item[abstract_key] = pubmed_details_list[0]["abstract"]
            yield item
