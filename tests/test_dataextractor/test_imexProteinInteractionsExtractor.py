import os
from logging.config import fileConfig
from unittest import TestCase

from ddt import ddt, data, unpack

from dataextractors.ImexProteinInteractionsExtractor import ImexProteinInteractionsExtractor


@ddt
class TestImexProteinInteractionsExtractor(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))

    @data(("data/human_13_negative.xml", 0)
        , ("data/human_01.xml", 4))
    @unpack
    def test_extract_protein_interaction_match_total(self, xmlfile, expected_total):
        # Arrange
        full_xml_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), xmlfile)
        sut = ImexProteinInteractionsExtractor(['phosphorylation'])

        # Act
        actual = list(sut.get_protein_interactions(full_xml_file_path))

        # Assert
        self.assertEqual(len(actual), expected_total)

    def test_extract_protein_interaction(self):
        # Arrange
        full_xml_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/human_01.xml")
        sut = ImexProteinInteractionsExtractor(['phosphorylation'])
        expected = [{'isNegative': 'false', 'participants': [{'uniprotid': 'Q16539-1', 'alias': [['q16539-1'], [
            'Mitogen-activated protein kinase 14'], ['MAPK14'], ['CSBP'], ['CSBP1'], ['CSBP2'], ['CSPB1'], ['MXI2'],
                                                                                                 ['SAPK2A'], [
                                                                                                     'Cytokine suppressive anti-inflammatory drug-binding protein'],
                                                                                                 ['MAP kinase MXI2'], [
                                                                                                     'MAX-interacting protein 2'],
                                                                                                 [
                                                                                                     'Mitogen-activated protein kinase p38 alpha'],
                                                                                                 [
                                                                                                     'Stress-activated protein kinase 2a']],
                                                              'alternative_uniprots': []}, {'uniprotid': 'P22736-1',
                                                                                            'alias': [['p22736-1'], [
                                                                                                'Nuclear receptor subfamily 4 group A member 1'],
                                                                                                      ['NR4A1'],
                                                                                                      ['GFRP1'],
                                                                                                      ['HMR'], ['NAK1'],
                                                                                                      [
                                                                                                          'Early response protein NAK1'],
                                                                                                      [
                                                                                                          'Nuclear hormone receptor NUR/77'],
                                                                                                      [
                                                                                                          'Orphan nuclear receptor HMR'],
                                                                                                      [
                                                                                                          'Orphan nuclear receptor TR3'],
                                                                                                      ['ST-59'], [
                                                                                                          'Testicular receptor 3']],
                                                                                            'alternative_uniprots': []}],
                     'pubmedId': '25822914',
                     'pubmedTitle': 'Impeding the interaction between Nur77 and p38 reduces LPS-induced inflammation.',
                     'interactionType': 'phosphorylation', 'interactionId': '2728358'}, {'isNegative': 'false',
                                                                                         'participants': [
                                                                                             {'uniprotid': 'P49841-2',
                                                                                              'alias': [['p49841-2'], [
                                                                                                  'Glycogen synthase kinase-3 beta'],
                                                                                                        ['GSK3B'], [
                                                                                                            'Serine/threonine-protein kinase GSK3B'],
                                                                                                        ['GSK-3beta2'],
                                                                                                        [
                                                                                                            'neuron-specific']],
                                                                                              'alternative_uniprots': []}],
                                                                                         'pubmedId': '25860027',
                                                                                         'pubmedTitle': 'GSK3β-Dzip1-Rab8 cascade regulates ciliogenesis after mitosis.',
                                                                                         'interactionType': 'phosphorylation',
                                                                                         'interactionId': '2728638'},
                    {'isNegative': 'false', 'participants': [{'uniprotid': 'Q8BMD2-1',
                                                              'alias': [['q8bmd2-1'], ['Zinc finger protein DZIP1'],
                                                                        ['Dzip1'], ['Kiaa0996'],
                                                                        ['DAZ-interacting protein 1 homolog']],
                                                              'alternative_uniprots': []}, {'uniprotid': 'P49841-2',
                                                                                            'alias': [['p49841-2'], [
                                                                                                'Glycogen synthase kinase-3 beta'],
                                                                                                      ['GSK3B'], [
                                                                                                          'Serine/threonine-protein kinase GSK3B'],
                                                                                                      ['GSK-3beta2'], [
                                                                                                          'neuron-specific']],
                                                                                            'alternative_uniprots': []}],
                     'pubmedId': '25860027',
                     'pubmedTitle': 'GSK3β-Dzip1-Rab8 cascade regulates ciliogenesis after mitosis.',
                     'interactionType': 'phosphorylation', 'interactionId': '2728655'}, {'isNegative': 'false',
                                                                                         'participants': [
                                                                                             {'uniprotid': 'Q8BMD2-1',
                                                                                              'alias': [['q8bmd2-1'], [
                                                                                                  'Zinc finger protein DZIP1'],
                                                                                                        ['Dzip1'],
                                                                                                        ['Kiaa0996'], [
                                                                                                            'DAZ-interacting protein 1 homolog']],
                                                                                              'alternative_uniprots': []},
                                                                                             {'uniprotid': 'P49841-2',
                                                                                              'alias': [['p49841-2'], [
                                                                                                  'Glycogen synthase kinase-3 beta'],
                                                                                                        ['GSK3B'], [
                                                                                                            'Serine/threonine-protein kinase GSK3B'],
                                                                                                        ['GSK-3beta2'],
                                                                                                        [
                                                                                                            'neuron-specific']],
                                                                                              'alternative_uniprots': []}],
                                                                                         'pubmedId': '25860027',
                                                                                         'pubmedTitle': 'GSK3β-Dzip1-Rab8 cascade regulates ciliogenesis after mitosis.',
                                                                                         'interactionType': 'phosphorylation',
                                                                                         'interactionId': '2728665'}]

        # Act
        actual = list(sut.get_protein_interactions(full_xml_file_path))

        # Assert
        self.assertEqual(expected, actual)
