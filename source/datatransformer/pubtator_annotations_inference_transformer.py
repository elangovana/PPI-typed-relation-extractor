import argparse
import glob
import itertools
import json
import logging
import os
import sys

from dataformatters.gnormplusPubtatorReader import GnormplusPubtatorReader
from datatransformer.ncbiGeneUniprotLocalDbMapper import NcbiGeneUniprotLocalDbMapper
from datatransformer.ncbiGeneUniprotMapper import NcbiGeneUniprotMapper
from datatransformer.textGeneNormaliser import TextGeneNormaliser


class PubtatorAnnotationsInferenceTransformer:

    def __init__(self, geneIdConverter=None, filter_self_relation=False):
        """
        Prepares inference
        :type interaction_types: list
        """
        self.filter_self_relation = filter_self_relation
        self.pubtator_annotations_reader = None
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
        result = {}

        for a in filter(lambda x: x["type"] == "Gene", annotations):
            uniprot_ids = self.geneIdConverter.convert(a["normalised_id"]).get(a["normalised_id"], None)

            # Add Mapping only when available
            if uniprot_ids is not None:
                # use just the first ID, even if there is more than mapping..
                uniprot_id = uniprot_ids[0]
                result[a["normalised_id"]] = uniprot_id
        return result

    def _reverse_uniprot_name_map(self, uniprot, gene_to_prot_map, annot):
        uniprot_matches = list(filter(lambda x: uniprot in gene_to_prot_map[x], gene_to_prot_map))

        if len(uniprot_matches) == 0: return None

        for a in annot:

            if a["normalised_id"] == uniprot_matches[0]:
                return a["name"]

    def load_directory(self, dir_path):
        files = glob.glob("{}/*.txt".format(dir_path))
        for input_file in files:
            self.logger.info("Processing file {}".format(input_file))
            yield (r for r in self.load_file(input_file))

    def load_directory_save(self, input_dir, destination_dir):
        files = glob.glob("{}/*.txt".format(input_dir))
        total = 0;
        for input_file in files:
            result = [r for r in self.load_file(input_file)]
            self.logger.info("Processed file {} with records {}".format(input_file, len(result)))
            total += len(result)
            if len(result) == 0:
                self.logger.info("No records generated")
                continue
            destination_file = os.path.join(destination_dir, "{}.json".format(os.path.basename(input_file)))
            with open(destination_file, "w") as fp:
                json.dump(result, fp)
        self.logger.info("Completed with {} files and {} records ".format(len(files), total))

    def parse(self, handle):
        for rec in self.pubtator_annotations_reader(handle):
            normalised_abstract, new_annotations = self.textGeneNormaliser(rec['text'], rec['annotations'])
            genes_map = self._get_genes(rec['annotations'])
            genes = list(genes_map.values())

            combinator = itertools.combinations_with_replacement(genes, 2)
            if self.filter_self_relation:
                combinator = itertools.combinations(genes, 2)

            for gene_pair in combinator:
                gene_pair = sorted(list(gene_pair))
                participant1_id = gene_pair[0]
                participant1_name = self._reverse_uniprot_name_map(participant1_id, genes_map, rec['annotations'])
                participant2_id = gene_pair[1]
                participant2_name = self._reverse_uniprot_name_map(participant2_id, genes_map, rec['annotations'])

                yield {'pubmedId': rec['id']
                    , 'participant1Id': participant1_id
                    , 'participant1Name': participant1_name
                    , 'participant2Id': participant2_id
                    , 'participant2Name': participant2_name
                    , 'abstract': rec['text']
                    , 'normalised_abstract': normalised_abstract
                    , 'annotations': rec['annotations']
                    , 'gene_to_uniprot_map': genes_map
                    , "normalised_abstract_annotations": new_annotations
                       }


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir",
                        help="The input dir containing annotated abstract files. The files should be *.txt formatted")
    parser.add_argument("outputdir", help="The output dir, this will contain json files..")
    parser.add_argument("idMappingDat", help="""The location of the idMapping dat file..
    You can obtain this file ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping. 
    This contains the ID mapping between UNIPROT and NCBI. We need this as GNormplus use NCBI gene id and we need the protein names.
    The dat file contains three columns, delimited by tab:
    - UniProtKB-AC
    - ID_type
    - ID
""")
    parser.add_argument("--filterselfrelations", help="To filter self relations", default=1, type=bool, choices={1, 0})

    args = parser.parse_args()
    with open(args.idMappingDat, "r") as h:
        geneIdconverter = NcbiGeneUniprotLocalDbMapper(h)
        obj = PubtatorAnnotationsInferenceTransformer(
            geneIdconverter, filter_self_relation=args.filterselfrelations)
        obj.load_directory_save(args.inputdir, args.outputdir)
