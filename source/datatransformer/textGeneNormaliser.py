from datatransformer.ncbiGeneUniprotMapper import NcbiGeneUniprotMapper


class TextGeneNormaliser:

    def __init__(self, geneIdConverter=None):
        self.geneIdConverter = geneIdConverter

    @property
    def geneIdConverter(self):
        self.__geneIdConverter__ = self.__geneIdConverter__ or NcbiGeneUniprotMapper()
        return self.__geneIdConverter__

    @geneIdConverter.setter
    def geneIdConverter(self, value):
        self.__geneIdConverter__ = value

    def __call__(self, text, annotations, preferred_uniprots=None, annotation_id_key='normalised_id'):

        offset = 0
        annotations.sort(key=lambda x: int(x['start']), reverse=False)
        preferred_uniprots = preferred_uniprots or {}

        new_annotations = []

        name_to_ncbi_map = {}
        for a in annotations:
            if a['type'].lower() != 'gene': continue
            name_to_ncbi_map[a['name']] = a[annotation_id_key]

        for a in annotations:
            if a['type'].lower() != 'gene': continue

            s = int(a['start']) + offset
            e = int(a['end']) + offset

            ncbi_id = a[annotation_id_key]

            # We might get more than one matching uniprot.
            # e.g. {'6850': ['P43405','A0A024R244','A0A024R273']}
            # or if no match then return the key as is, [ncbi_id]
            uniprots = self.geneIdConverter.convert(ncbi_id).get(ncbi_id, [ncbi_id])

            # # By default, use that else just pick the first one
            uniprot = uniprots[0]

            for p in preferred_uniprots:
                if p in uniprots:
                    uniprot = p
                    break

            text = text[:s] + uniprot + text[e:]
            offset += len(uniprot) - (e - s)

            new_annotations.append({
                "charOffset" : s,
                "len" : len(uniprot),
                "text" : uniprot
            })
        return text, new_annotations
