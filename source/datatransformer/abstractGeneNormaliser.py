"""
Replaces the gene name in abstract with the uniprot number
"""

from datatransformer.ncbiGeneUniprotMapper import NcbiGeneUniprotMapper


class AbstractGeneNormaliser:

    def __init__(self, pubmed_annotations: iter, geneIdConverter=None, key_func=None, abstract_func=None,
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
        self.geneIdConverter = geneIdConverter

    @property
    def geneIdConverter(self):
        self.__geneIdConverter__ = self.__geneIdConverter__ or NcbiGeneUniprotMapper()
        return self.__geneIdConverter__

    @geneIdConverter.setter
    def geneIdConverter(self, value):
        self.__geneIdConverter__ = value

    def transform(self, df):

        annotations_dict = self._construct_dict()
        df['normalised_abstract'] = df.apply(lambda r: annotations_dict[r['pubmedId']], axis=1)

        return df

    def _construct_dict(self):
        result = {}
        for r in self.pubmed_annotations:
            key = self.key_func(r)
            abstract = self.abstract_func(r)
            annotations = self.annotations_func(r)

            normalised_abstract = self._normalise_abstract(annotations, abstract)

            result[key] = normalised_abstract
        return result

    def _normalise_abstract(self, annotations, abstract):
        offset = 0
        annotations.sort(key=lambda x: int(x['start']), reverse=False)
        for a in annotations:
            if a['type'].lower() != 'gene': continue

            s = int(a['start']) + offset
            e = int(a['end']) + offset

            ncbi_id = a['normalised_id']

            uniprot = self.geneIdConverter.convert(ncbi_id).get(ncbi_id, ncbi_id)

            abstract = abstract[:s] + uniprot + abstract[e:]
            offset += len(uniprot) - (e - s)
        return abstract
