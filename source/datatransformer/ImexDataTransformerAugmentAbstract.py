import logging
from multiprocessing.dummy import Pool

from dataextractors.PubmedAbstractExtractor import PubmedAbstractExtractor


class ImexDataTransformerAugmentAbstract:
    def __init__(self, id_key: str = "pubmedId",
                 abstract_key: str = "pubmedabstract", pool_size=None):
        """
        :param id_key: The key in the input identifying the pubmedid used to extract the abstract
        :param abstract_key: The key to use for the resulting abstract
        """
        self.abstract_key = abstract_key
        self.id_key = id_key
        self.pubmed_extractor = None
        self.pool_size = 2
        self.logger.info("Using pool size of {}".format(self.pool_size))

    @property
    def pubmed_extractor(self):
        self.__pubmed_extractor__ = self.__pubmed_extractor__ or PubmedAbstractExtractor()
        return self.__pubmed_extractor__

    @pubmed_extractor.setter
    def pubmed_extractor(self, value):
        self.__pubmed_extractor__ = value

    @property
    def logger(self):
        return logging.getLogger(__name__)

    #     def transform(self, input_iterable: iter) -> iter:
    #         """
    # Adds the pubmed abstract to each item based on the pubmed id
    #         :param input_iterable: A iterable list of dictionary
    #          """
    #         for item in input_iterable:
    #             pubmed_details_list = self.pubmed_extractor.extract_abstract_by_pubmedid([item[self.id_key]])
    #
    #             if len(pubmed_details_list) == 0:
    #                 self.logger.warning("Could not find abstract for record {}".format(item[self.id_key]))
    #                 yield None
    #
    #             # expect only one record in the array
    #             assert len(pubmed_details_list) == 1
    #             item[self.abstract_key] = pubmed_details_list[0]["abstract"]
    #             yield item

    def transform(self, input_iterable: iter) -> iter:
        """
Adds the pubmed abstract to each item based on the pubmed id
        :param input_iterable: A iterable list of dictionary
         """

        p = Pool(self.pool_size)
        result = p.map(self._get_pubmed_abstract, input_iterable)

        # remove nones from the list when the abstract could not be found
        result = filter(None, result)

        return result

    def _get_pubmed_abstract(self, item):
        pubmed_details_list = self.pubmed_extractor.extract_abstract_by_pubmedid([item[self.id_key]])
        if len(pubmed_details_list) == 0:
            self.logger.warning("Could not find abstract for record {}".format(item[self.id_key]))
            return

        # expect only one record in the array
        assert len(pubmed_details_list) == 1, "Expecting just 1,  but found {}".format(len(pubmed_details_list))
        item[self.abstract_key] = pubmed_details_list[0]["abstract"]
        return item
