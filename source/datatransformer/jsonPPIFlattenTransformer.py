import pandas as pd
import json
import copy


class IntactJsonPpiFlattenTransformer:
    """
    Flattens a dataframe with participants array ..
            interactionId	interactionType	isNegative	participants	pubmedId	pubmedTitle	pubmedabstract
             1	        phosphorylation	True	    [{'uniprotid':
             'Q8WUB1', 'alias': [['kc1g2_human'], ['Casein kinase I isoform gamma-2'], ['CK1G2'], ['CSNK1G2']]},
             {'uniprotid': 'Q6ZS50', 'alias': [['lrrk2_human'], ['Leucine-rich repeat serine/threonine-protein kinase
             2'], ['LRRK2'], ['PARK8'], ['Dardarin']]}]	25605870	None	NLRP3 is the most crucial member of the
             NLR family, as it detects the existence of pathogen invasion and self-derived molecules associated with
             cellular damage. Several studies have reported that excessive NLRP3 inflammasome-mediated caspase-1
             activation is a key factor in the development of diseases. Recent studies have reported that Syk is
             involved in pathogen-induced NLRP3 inflammasome activation; however, the detailed mechanism linking Syk
             to NLRP3 inflammasome remains unclear. In this study, we showed that Syk mediates NLRP3 stimuli-induced
             processing of procaspase-1 and the consequent activation of caspase-1. Moreover, the kinase activity of
             Syk is required to potentiate caspase-1 activation in a reconstituted NLRP3 inflammasome system in
             HEK293T cells. The adaptor protein ASC bridges NLRP3 with the effector protein caspase-1. Herein,
             we find that Syk can associate directly with ASC and NLRP3 by its kinase domain but interact indirectly
             with procaspase-1. Syk can phosphorylate ASC at Y146 and Y187 residues, and the phosphorylation of both
             residues is critical to enhance ASC oligomerization and the recruitment of procaspase-1. Together,
             our results reveal a new molecular pathway through which Syk promotes NLRP3 inflammasome formation,
             resulting from the phosphorylation of ASC. Thus, the control of Syk activity might be effective to
             modulate NLRP3 inflammasome activation and treat NLRP3-related immune diseases.	True
    """

    def __init__(self, ):
        pass

    def transform(self, dataframe):

        json_data = json.loads(dataframe.to_json(orient='records'))
        resulting_json = []
        for r in json_data:
            participants = r["participants"]

            # Case self relation
            if len(participants) == 1:
                record = self.construct_flatrecord(r, participants[0], participants[0])
                resulting_json.append(record)
                continue

            # Case more than one partcipate, create pairwise relationship
            for i in range(len(participants)):
                s = participants[i]
                for j in range(i + 1, len(participants)):
                    t = participants[j]
                    record = self.construct_flatrecord(r, s, t)
                    resulting_json.append(record)

        return pd.DataFrame(resulting_json)

    def construct_flatrecord(self, record_to_copy_from, source_participant, target_participant):
        """
               Flattens participants
               [{
                   'uniprotid': 'Q8WUB1',
                   'alias': [
                     ['kc1g2_human'],
                     ['Casein kinase I isoform gamma-2'],
                     ['CK1G2'],
                     ['CSNK1G2']
                   ]
                 }]
           """
        record = copy.deepcopy(record_to_copy_from)
        record.pop('participants', None)
        record["sourceUniprot"] = source_participant["uniprotid"]
        record["destUnitprot"] = target_participant["uniprotid"]
        record["sourceAlias"] = source_participant["alias"]
        record["destAlias"] = target_participant["alias"]
        return record
