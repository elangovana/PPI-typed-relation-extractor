import logging
from xml.etree import ElementTree

from bioservices import UniProt


class ImexProteinInteractionsExtractor:

    def __init__(self, interactionlist=None):

        if interactionlist is None:
            interactionlist = ["phosphorylation"]
        self.interactionlist = interactionlist

        self.namespaces = {'df': 'http://psi.hupo.org/mi/mif'}  #
        self._logger = logging.getLogger(__name__)

        self.u = UniProt(verbose=False)
        self._cache_kegg_entry_uniprots = {}

    def get_protein_interactions(self, xmlfile):
        self._logger.info("Extracting PPIs for file {} ".format(xmlfile))

        with open(xmlfile, "r") as handle:

            for entry in self._iter_elements_by_name(handle, "df:entry", self.namespaces):

                ele_interaction_list = entry.findall("df:interactionList/df:interaction", self.namespaces)
                for ele_interaction in ele_interaction_list:
                    interaction_type = ele_interaction.find("df:interactionType/df:names/df:shortLabel",
                                                            self.namespaces).text
                    if interaction_type not in self.interactionlist:
                        continue
                    i = 0
                    interaction_id = ele_interaction.attrib["id"]
                    experiment_ref_id = ele_interaction.find("df:experimentList/df:experimentRef", self.namespaces).text
                    pubmed_id, title = self.get_pubmed_id(entry, experiment_ref_id)

                    is_negative = ele_interaction.find("df:negative", self.namespaces).text

                    participants = []
                    for ele_participant in ele_interaction.findall("df:participantList/df:participant",
                                                                   self.namespaces):
                        interfactor_ref_id = ele_participant.find("df:interactorRef", self.namespaces).text

                        uniprotid, alias_list, alternative_uniprot = self.get_interactor_details(entry,
                                                                                                 interfactor_ref_id)
                        participants.append(
                            {"uniprotid": uniprotid, "alias": alias_list, 'alternative_uniprots': alternative_uniprot})

                        i = i + 1

                    yield {
                        "isNegative": is_negative
                        , "participants": participants
                        , "pubmedId": pubmed_id
                        , "pubmedTitle": title
                        , "interactionType": interaction_type
                        , "interactionId": interaction_id

                    }

    def _iter_elements_by_name(self, handle, name, namespace):
        events = ElementTree.iterparse(handle, events=("start", "end"))
        _, root = next(events)  # Grab the root element.

        expanded_name = name
        # If name has the namespace, expand it
        if name.index(":") >= 0:
            local_name = name[name.index(":") + 1:]
            namespace_short_name = name[:name.index(":")]
            expanded_name = "{{{}}}{}".format(namespace[namespace_short_name], local_name)

        for event, elem in events:

            if event == "end" and elem.tag == expanded_name:
                yield elem
                elem.clear()

    def get_interactor_details(self, entry, interfactor_ref_id):
        interactor_xpath = "df:interactorList/df:interactor[@id='{}']".format(interfactor_ref_id)
        ele_interactor = entry.find(interactor_xpath, self.namespaces)
        ele_unitprot = ele_interactor.find("df:xref/df:primaryRef[@db='{}']".format("uniprotkb"), self.namespaces)
        alias = []
        result = (None, None, None)
        for e in ele_interactor.findall("df:names//*", self.namespaces):
            alias.append([e.text])

        alternative_uniprots = []
        for e in ele_interactor.findall("df:xref/df:secondaryRef[@db='{}']".format("uniprotkb"), self.namespaces):
            alternative_uniprots.append(e.attrib['id'])

        # If no primary ref, try the secondary ref
        if ele_unitprot is None:
            ele_unitprot = ele_interactor.find("df:xref/df:secondaryRef[@db='{}']".format("uniprotkb"), self.namespaces)
        if ele_unitprot is not None:
            result = ele_unitprot.attrib['id'], alias, alternative_uniprots

        return result

    def get_pubmed_id(self, entry, experiment_ref_id):
        ele_experiment = entry.find("df:experimentList/df:experimentDescription[@id='{}']".format(experiment_ref_id),
                                    self.namespaces)
        ele_primary_ref = ele_experiment.find("df:bibref/df:xref/df:primaryRef[@db='{}']".format("pubmed"),
                                              self.namespaces)

        if ele_primary_ref is not None:
            title = None
            ele_title = ele_experiment.find("df:attributeList/df:attribute[@name='title']", self.namespaces)
            if ele_title is not None:
                title = ele_title.text

            return (ele_primary_ref.attrib["id"], title)
        return None
