"""
Replaces the gene name in abstract with the uniprot number
"""

from datatransformer.textGeneNormaliser import TextGeneNormaliser


class AbstractGeneNormaliser:

    def __init__(self, pubmed_annotations: iter, key_func=None, abstract_func=None,
                 annotations_func=None):
        """
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
        self.annotations_func = annotations_func or (lambda x: x['annotations'])
        self.abstract_func = abstract_func or (lambda x: x['text'])
        self.key_func = key_func or (lambda x: x['id'])
        self.pubmed_annotations = pubmed_annotations
        self.textGeneNormaliser = None

    @property
    def textGeneNormaliser(self):
        self.__textGeneNormaliser__ = self.__textGeneNormaliser__ or TextGeneNormaliser()
        return self.__textGeneNormaliser__

    @textGeneNormaliser.setter
    def textGeneNormaliser(self, value):
        self.__textGeneNormaliser__ = value

    def transform(self, df):
        annotations_dict = self._construct_dict()
        df['normalised_abstract'] = df.apply(
            lambda r: self._normalise_abstract(annotations_dict[r['pubmedId']]["annotations"],
                                               annotations_dict[r['pubmedId']]["abstract"],
                                               {r['participant1Id']: r['participant1Alias'],
                                                r['participant2Id']: r['participant2Alias']}),
            axis=1)

        # Also add annotations to the data frame..
        df['annotations'] = df.apply(
            lambda r: annotations_dict[r['pubmedId']]["annotations"],
            axis=1)

        df['annotations_abstract'] = df.apply(
            lambda r: annotations_dict[r['pubmedId']]["abstract"],
            axis=1)

        return df

    def _construct_dict(self):
        result = {}
        for r in self.pubmed_annotations:
            key = self.key_func(r)
            abstract = self.abstract_func(r)
            annotations = self.annotations_func(r)

            result[key] = {"abstract": abstract, "annotations": annotations}
        return result

    def _normalise_abstract(self, annotations, abstract, preferred_uniprots=None):
        abstract = self.textGeneNormaliser(abstract, annotations, preferred_uniprots)
        return abstract
