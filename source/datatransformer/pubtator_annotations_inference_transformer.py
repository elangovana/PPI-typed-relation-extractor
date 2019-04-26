import glob
import itertools
import logging

from dataformatters.gnormplusPubtatorReader import GnormplusPubtatorReader
from datatransformer.ncbiGeneUniprotMapper import NcbiGeneUniprotMapper
from datatransformer.textGeneNormaliser import TextGeneNormaliser


class PubtatorAnnotationsInferenceTransformer:

    def __init__(self, interaction_types=None, geneIdConverter=None):
        """
        Prepares inference for these interaction types, by default uses 'phosphorylation'
        :type interaction_types: list
        """
        self.pubtator_annotations_reader = None
        self.interaction_types = interaction_types or ['phosphorylation']
        self.geneIdConverter = geneIdConverter
        self.textGeneNormaliser = None

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def pubtator_annotations_reader(self):
        self.__pubtator_annotations_reader__ = self.__pubtator_annotations_reader__ or GnormplusPubtatorReader()
        return self.__pubtator_annotations_reader__

    @property
    def geneIdConverter(self):
        """
Convert Ncbi geneId to uniprot
        :return:
        """
        self.__geneIdConverter__ = self.__geneIdConverter__ or NcbiGeneUniprotMapper()
        return self.__geneIdConverter__

    @geneIdConverter.setter
    def geneIdConverter(self, value):
        self.__geneIdConverter__ = value

    @property
    def textGeneNormaliser(self):
        """
Convert Ncbi geneId to uniprot
        :return:
        """
        if self.__textGeneNormaliser__ is None:
            self.__textGeneNormaliser__ = TextGeneNormaliser()
            self.__textGeneNormaliser__.geneIdConverter = self.geneIdConverter

        return self.__textGeneNormaliser__

    @textGeneNormaliser.setter
    def textGeneNormaliser(self, value):
        self.__textGeneNormaliser__ = value

    @pubtator_annotations_reader.setter
    def pubtator_annotations_reader(self, value):
        self.__pubtator_annotations_reader__ = value

    def load_file(self, input_file_path):
        with open(input_file_path, "r") as i:
            yield from self.parse(i)

    def _get_genes(self, annotations):
        result = set()
        for a in filter(lambda x: x["type"] == "Gene", annotations):
            uniprot_id = self.geneIdConverter.convert(a["normalised_id"]).get(a["normalised_id"], [a["normalised_id"]])
            result = result.union(uniprot_id)
        return result

    def load_directory(self, dir_path):
        files = glob.glob("{}/*.txt".format(dir_path))
        for input_file in files:
            self.logger.info("Processing file {}".format(input_file))
            yield (r for r in self.load_file(input_file))

    def parse(self, handle):
        for rec in self.pubtator_annotations_reader(handle):
            normalised_abstract = self.textGeneNormaliser(rec['text'], rec['annotations'])
            genes = self._get_genes(rec['annotations'])

            for gene_pair in itertools.combinations_with_replacement(genes, 2):
                for interaction_type in self.interaction_types:
                    gene_pair = sorted(list(gene_pair))
                    yield {'pubmedId': rec['id']
                        , 'interactionType': interaction_type
                        , 'participant1Id': gene_pair[0]
                        , 'participant2Id': gene_pair[1]
                        , 'abstract': rec['text']
                        , 'normalised_abstract': normalised_abstract
                           }
