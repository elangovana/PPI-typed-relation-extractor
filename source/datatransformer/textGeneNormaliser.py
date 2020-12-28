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

        name_to_ncbi_map = {}
        for a in annotations:
            if a['type'].lower() != 'gene': continue
            name_to_ncbi_map[a['name']] = a[annotation_id_key]

        # alternative_ncbi_uniprot = {}
        # for g_in_anno, ncbi in name_to_ncbi_map.items():
        #     for u, aliases in preferred_uniprots.items():
        #         flatened_alias = [i.lower() for i in reduce(operator.concat, aliases)]
        #
        #         if g_in_anno.lower() in flatened_alias:
        #             alternative_ncbi_uniprot[ncbi] = u

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

            match = False
            for p in preferred_uniprots:
                if p in uniprots:
                    uniprot = p
                    match = True
                    break

            # # Some of the uniprots dont match.. so try match with alias
            # if not match:
            #     uniprot = alternative_ncbi_uniprot.get(ncbi_id, uniprot)

            text = text[:s] + uniprot + text[e:]
            offset += len(uniprot) - (e - s)
        return text
