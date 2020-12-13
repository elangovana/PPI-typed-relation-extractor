"""
Replaces the gene name in abstract with the uniprot number
"""

from datatransformer.textGeneNormaliser import TextGeneNormaliser


class AbstractGeneNormaliser:

    def __init__(self, pubmed_annotations: iter, key_func=None, abstract_func=None,
                 annotations_func=None, field_name_prefix=''):
        """
        :param field_name_prefix: The name of the output dataframe field value
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
        annotations_dict = self._construct_dict()
        df[f'{self.field_name_prefix}normalised_abstract'] = df.apply(
            lambda r: self._normalise_abstract(annotations_dict[r['pubmedId']]["annotations"],
                                               annotations_dict[r['pubmedId']]["abstract"],
                                               {r['participant1Id']: r['participant1Alias'],
                                                r['participant2Id']: r['participant2Alias']}),
            axis=1)

        # Also add annotations to the data frame..
        df[f'{self.field_name_prefix}annotations'] = df.apply(
            lambda r: annotations_dict[r['pubmedId']]["annotations"],
            axis=1)

        df[f'{self.field_name_prefix}annotations_abstract'] = df.apply(
            lambda r: annotations_dict[r['pubmedId']]["abstract"],
            axis=1)

        df[f'{self.field_name_prefix}num_unique_gene_normalised_id'] = df["annotations"].apply(
            self._count_unique_gene_id_mentions)

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
