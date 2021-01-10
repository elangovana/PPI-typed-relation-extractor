import copy
import logging
import uuid

import pandas as pd

from datatransformer.ncbiGeneUniprotMapper import NcbiGeneUniprotMapper

"""
Adds negative samples of entities (Uniprot numbers) found in annotations in a entity recognition step. 
"""


class GnormplusNegativeSamplesAugmentor:

    def __init__(self, pubmed_annotations: iter, geneIdConverter=None, include_self_relations=False,
                 max_negative_per_pubmed=None):
        """
         :param include_self_relations: Set this to true if you want self-relations to be included as negative samples
         :param pubmed_annotations: The pubmed anntations looks like this is an array of dict
          [{'id': '19167335', 'type': 'a',
                      'text': 'Protein tyrosine.. phosphatases',
                      'annotations': [
                          {'start': '167', 'end': '170', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'},
                          {'start': '287', 'end': '290', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'},
                         ]}
                         }]
         :param geneIdConverter:
         """
        self.max_negative_per_pubmed = max_negative_per_pubmed
        self.include_self_relations = include_self_relations
        self.pubmed_annotations = pubmed_annotations
        self.geneIdConverter = geneIdConverter

    @property
    def geneIdConverter(self):
        self.__geneIdConverter__ = self.__geneIdConverter__ or NcbiGeneUniprotMapper()
        return self.__geneIdConverter__

    @geneIdConverter.setter
    def geneIdConverter(self, value):
        self.__geneIdConverter__ = value

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def transform(self, input_df):
        """

        :param input_df: A pandas dataframe like object with columns atleast these columns [interactionId, isValid, pubmedId, particpant1Id, particpant2Id] .
        """

        data_fake = pd.DataFrame(columns=input_df.columns)

        pubmedid_entities_map = self._construct_pubmed_entities_map()

        for pubmedid in input_df['pubmedId'].unique():
            existing_participants = set()

            all_possible_participants = pubmedid_entities_map[pubmedid]["participant_pairs"]
            alias_map = pubmedid_entities_map[pubmedid]["aliases_map"]

            for p_pair in input_df.query("pubmedId == '{}'".format(pubmedid))[
                ["participant1Id", "participant2Id"]].itertuples():
                participants = frozenset([p_pair[1], p_pair[2]])
                existing_participants.add(participants)

            new_participants = all_possible_participants - existing_participants
            new_participants = sorted(new_participants, key=lambda x: "#".join(sorted(list(x))))

            template_record = input_df[input_df.pubmedId == pubmedid].iloc[0]

            # Only use participants if only frozen set has more than one participants
            self_filter = lambda x: len(x) > 1
            if self.include_self_relations:
                # If include self relations, no filter
                self_filter = lambda x: True

            for i, p in enumerate(filter(self_filter, new_participants)):

                l = sorted(list(p))
                p1 = l[0]
                p2 = l[1]

                record = copy.deepcopy(template_record)
                record["isValid"] = False
                record["participant1Id"] = p1
                record["participant2Id"] = p2
                record["participant1Alias"] = sorted(list(alias_map[p1]))
                record["participant2Alias"] = sorted(list(alias_map[p2]))
                record["interactionId"] = record["interactionId"] + "_" + str(
                    uuid.uuid4()) + "_" + "fake_annot"

                data_fake = data_fake.append(record)

                if self.max_negative_per_pubmed is not None and self.max_negative_per_pubmed == i:
                    break

        # Data with negative samples
        data_fake = data_fake.append(input_df, ignore_index=True)
        return data_fake

    def _construct_pubmed_entities_map(self):
        map_gene_ids = {}

        for annotation in self.pubmed_annotations:
            # Get unique mentions of converted gene ids
            converted_gene_ids = set()
            gene_id_alias_map = {}
            for g in filter(lambda v: v['type'] == 'Gene', annotation['annotations']):
                normalised_id = g['normalised_id']
                coverted_map_list = self.geneIdConverter.convert(normalised_id)
                if normalised_id in coverted_map_list:
                    geneid = coverted_map_list[normalised_id][0]
                else:
                    continue
                converted_gene_ids.add(geneid)

                # also create a map of geneid: alias so it is easier to refer back to the name used in the text
                gene_id_alias_map[geneid] = gene_id_alias_map.get(geneid, set()).union({g['name']})

            # construct protein pairs
            converted_gene_ids = list(converted_gene_ids)
            participant_pairs = set()
            for i in range(len(converted_gene_ids)):
                for j in range(i, len(converted_gene_ids)):
                    participants = frozenset([converted_gene_ids[i], converted_gene_ids[j]])
                    participant_pairs.add(participants)

            map_gene_ids[annotation["id"]] = {"participant_pairs": participant_pairs, "aliases_map": gene_id_alias_map}

        return map_gene_ids
