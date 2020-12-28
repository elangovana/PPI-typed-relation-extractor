"""
Replaces the gene name in abstract with the uniprot number
"""
import logging

from datatransformer.textGeneNormaliser import TextGeneNormaliser


class AbstractGeneNormaliser:

    def __init__(self, pubmed_annotations: iter, key_func=None, abstract_func=None,
                 annotations_func=None, field_name_prefix=''):
        """
        :param field_name_prefix: The prefix to use of the output fields in the dataframe
        :param annotations_func: A function to obtain the annotations value.
        :param key_func: A function to obtain the key value. e.g. (lambda x: x['id'])
        :param abstract_func: A function to obtain the abstract value.
        :param include_self_relations: Set this to true if you want self-relations to be included as negative samples
        :param pubmed_annotations: The pubmed anntations looks like this is an array of dict
         [{'id': '19167335', 'type': 'a',
                     'text': 'Protein tyrosine.. phosphatases',
                     'annotations': [
                         {'start': '167', 'end': '170', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'},
                         {'start': '287', 'end': '290', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'},
                        ]}
                        }]
        :param geneIdConverter:  Converter NCBI genes to the Uniprot
        """
        self._annotations_func = annotations_func or (lambda x: x['annotations'])
        self._abstract_func = abstract_func or (lambda x: x['text'])
        self._key_func = key_func or (lambda x: x['id'])
        self._pubmed_annotations = pubmed_annotations

        #
        self._text_gene_normaliser = TextGeneNormaliser()
        self._field_name_prefix = field_name_prefix

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    @property
    def text_gene_normaliser(self):
        return self._text_gene_normaliser

    @text_gene_normaliser.setter
    def text_gene_normaliser(self, value):
        self._text_gene_normaliser = value

    @property
    def field_name_prefix(self):
        return self._field_name_prefix

    @field_name_prefix.setter
    def field_name_prefix(self, value):
        self._field_name_prefix = value

    def transform(self, df):
        self._logger.info("Starting transformation..")

        annotations_dict = self._construct_dict()
        df[f'{self.field_name_prefix}normalised_abstract'] = df.apply(
            lambda r: self._normalise_abstract(annotations_dict[r['pubmedId']]["annotations"],
                                               annotations_dict[r['pubmedId']]["abstract"],
                                               {r['participant1Id']: r['participant1Alias'],
                                                r['participant2Id']: r['participant2Alias']}),
            axis=1)

        self._logger.info("Completed normalised abstract...")


        # Also add annotations to the data frame..
        self._logger.info("Adding annotations ...")
        annotations_field = f'{self.field_name_prefix}annotations'
        df[annotations_field] = df.apply(
            lambda r: annotations_dict[r['pubmedId']]["annotations"],
            axis=1)

        self._logger.info("Adding annotations_abstract...")
        df[f'{self.field_name_prefix}annotations_abstract'] = df.apply(
            lambda r: annotations_dict[r['pubmedId']]["abstract"],
            axis=1)

        self._logger.info("Adding num_unique_gene_normalised_id...")
        df[f'{self.field_name_prefix}num_unique_gene_normalised_id'] = df[annotations_field].apply(
            self._count_unique_gene_id_mentions)

        self._logger.info("Adding num_gene_normalised_id...")
        df[f'{self.field_name_prefix}num_gene_normalised_id'] = df[annotations_field].apply(
            self._count_gene_id_mentions)

        self._logger.info("Gene Id links...")
        df[f'{self.field_name_prefix}gene_to_uniprot_map'] = df[annotations_field].apply(
            self._gene_id_uniprot_map)

        self._logger.info("Completed transformation")


        return df

    def _construct_dict(self):
        result = {}
        for r in self._pubmed_annotations:
            key = self._key_func(r)
            abstract = self._abstract_func(r)
            annotations = self._annotations_func(r)

            result[key] = {"abstract": abstract, "annotations": annotations}
        return result

    def _normalise_abstract(self, annotations, abstract, preferred_uniprots=None):
        abstract = self.text_gene_normaliser(abstract, annotations, preferred_uniprots)
        return abstract

    def _count_unique_gene_id_mentions(self, annotations):
        """
        Returns the count of unique gene mentions ..
        :param annotations:
        [{'start': '0', 'end': '5', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'},
         {'start': '206', 'end': '211', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'}]
        :return:
        """
        unique_genes = set()

        for anno in annotations:
            # If not a gene annotation skip, as it could be species etc
            if anno["type"] != 'Gene': continue

            unique_genes = unique_genes.union([anno['normalised_id']])

        return len(unique_genes)

    def _gene_id_uniprot_map(self, annotations):
        """
        Maps the normalised id to  uniprot
        :param annotations:
        [{'start': '0', 'end': '5', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'},
         {'start': '206', 'end': '211', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'}]
        :return:
        """
        unique_genes = set()

        for anno in annotations:
            # If not a gene annotation skip, as it could be species etc
            if anno["type"] != 'Gene': continue

            unique_genes = unique_genes.union([anno['normalised_id']])

        result = {}
        for g in list(unique_genes):
            result[g] = self.text_gene_normaliser.geneIdConverter.convert(g).get(g, [])

        return result

    def _count_gene_id_mentions(self, annotations):
        """
        Returns the count of unique gene mentions ..
        :param annotations:
        [{'start': '0', 'end': '5', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'},
         {'start': '206', 'end': '211', 'name': 'NLRP3', 'type': 'Gene', 'normalised_id': '114548'}]
        :return:
        """

        gene_mentions = list(filter(lambda anno: anno["type"] == 'Gene', annotations))

        return len(gene_mentions)
