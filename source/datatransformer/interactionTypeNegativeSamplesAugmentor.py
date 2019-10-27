import copy
import logging
import random
import uuid

import pandas as pd

"""
Adds negative samples by adding false interaction types
"""


class InteractionTypeNegativeSamplesAugmentor:

    def __init__(self, seed=777, max_negative_per_pubmed=None):

        self.max_negative_per_pubmed = max_negative_per_pubmed
        self.seed = seed

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def transform(self, input_df):
        """

        :param input_df: A pandas dataframe like object with columns atleast these columns [interactionId, isNegative, pubmedId, interactionType] .
        """

        interaction_types = input_df['interactionType'].unique()
        tot_interactions = len(interaction_types)
        self.logger.debug("Interactions types found in the dataset {}".format(interaction_types))

        data_fake = pd.DataFrame(columns=input_df.columns)
        for pubmedid in input_df['pubmedId'].unique():
            # input records matching the given pubmed
            input_records_pubmed = input_df[input_df.pubmedId == pubmedid]
            pubmed_pos_relations = input_records_pubmed.query('isValid == True')

            # interaction types associated with current pubmed
            existing_interaction_types = input_records_pubmed['interactionType'].unique()

            fake_interactions = set(interaction_types) - set(existing_interaction_types)

            # Fix the max number of fake interactions to max_negative_per_pubmed
            max_negative_per_pubmed = self.max_negative_per_pubmed
            if self.max_negative_per_pubmed is None:
                max_negative_per_pubmed = len(fake_interactions)
            fake_interactions = random.sample(fake_interactions,
                                              min(max_negative_per_pubmed, len(fake_interactions)))

            sample_size = tot_interactions if len(pubmed_pos_relations) >= tot_interactions else len(
                pubmed_pos_relations)
            # get tep
            template_records = pubmed_pos_relations.sample(n=sample_size, random_state=self.seed)

            i = 0
            for p in fake_interactions:
                i = (i + 1) % len(template_records)
                template_record = template_records.iloc[i]
                record = copy.deepcopy(template_record)
                record["isValid"] = False
                record["interactionType"] = p
                record["interactionId"] = record["interactionId"] + "_" + str(
                    uuid.uuid4()) + "_" + "fake_interaction"

                data_fake = data_fake.append(record)

        # Data with negative samples
        data_fake = data_fake.append(input_df, ignore_index=True)
        return data_fake
