import functools
import json
import operator
import os
import tempfile
from io import StringIO
from unittest import TestCase
from unittest.mock import MagicMock

from datatransformer.pubtator_annotations_inference_transformer import PubtatorAnnotationsInferenceTransformer


class TestSitPubtatorAnnotationsInferenceTransformer(TestCase):
    def test_parse(self):
        # Arrange
        sut = PubtatorAnnotationsInferenceTransformer(interaction_types=['phosphorylation'])
        input = StringIO("""19167335|a|Protein tyrosine phosphatases (PTPs) play a critical role in regulating cellular functions by selectively dephosphorylating their substrates. Here we present 22 human PTP crystal structures that, together with prior structural knowledge, enable a comprehensive analysis of the classical PTP family. Despite their largely conserved fold, surface properties of PTPs are strikingly diverse. A potential secondary substrate-binding pocket is frequently found in phosphatases, and this has implications for both substrate recognition and development of selective inhibitors. Structural comparison identified four diverse catalytic loop (WPD) conformations and suggested a mechanism for loop closure. Enzymatic assays revealed vast differences in PTP catalytic activity and identified PTPD1, PTPD2, and HDPTP as catalytically inert protein phosphatases. We propose a "head-to-toe" dimerization model for RPTPgamma/zeta that is distinct from the "inhibitory wedge" model and that provides a molecular basis for inhibitory regulation. This phosphatome resource gives an expanded insight into intrafamily PTP diversity, catalytic activity, substrate recognition, and autoregulatory self-association.
19167335	167	170	PTP	Gene	10076
19167335	779	784	PTPD1	Gene	11099

25260751|a|Unlike the other MAP3Ks, MEKK1 (encoded by Map3k1) contains a PHD motif. To understand the role of this motif, we have created a knockin mutant of mouse Map3k1 (Map3k1(m) (PHD)) with an inactive PHD motif. Map3k1(m) (PHD) ES cells demonstrate that the MEKK1 PHD controls p38 and JNK activation during TGF-b, EGF and microtubule disruption signalling, but does not affect MAPK responses to hyperosmotic stress. Protein microarray profiling identified the adaptor TAB1 as a PHD substrate, and TGF-b- or EGF-stimulated Map3k1(m) (PHD) ES cells exhibit defective non-canonical ubiquitination of MEKK1 and TAB1. The MEKK1 PHD binds and mediates the transfer of Lys63-linked poly-Ub, using the conjugating enzyme UBE2N, onto TAB1 to regulate TAK1 and MAPK activation by TGF-b and EGF. Both the MEKK1 PHD and TAB1 are critical for ES-cell differentiation and tumourigenesis. Map3k1(m) (PHD) (/+) mice exhibit aberrant cardiac tissue, B-cell development, testis and T-cell signalling. 
25260751	25	30	MEKK1	Gene	26401

""")

        mock_text_nomaliser = MagicMock()
        mock_text_nomaliser.return_value = "Normalisedtext.."
        sut.textGeneNormaliser = mock_text_nomaliser

        mock_gene_converter = MagicMock()
        mock_gene_converter.convert.side_effect = lambda x: {x: ["Q{}".format(x)]}
        sut.geneIdConverter = mock_gene_converter

        expected = [

            {"pubmedId": "19167335"
                , "interactionType": "phosphorylation"
                , "participant1Id": "Q10076"
                , "participant2Id": "Q10076"
                ,
             "abstract": """Protein tyrosine phosphatases (PTPs) play a critical role in regulating cellular functions by selectively dephosphorylating their substrates. Here we present 22 human PTP crystal structures that, together with prior structural knowledge, enable a comprehensive analysis of the classical PTP family. Despite their largely conserved fold, surface properties of PTPs are strikingly diverse. A potential secondary substrate-binding pocket is frequently found in phosphatases, and this has implications for both substrate recognition and development of selective inhibitors. Structural comparison identified four diverse catalytic loop (WPD) conformations and suggested a mechanism for loop closure. Enzymatic assays revealed vast differences in PTP catalytic activity and identified PTPD1, PTPD2, and HDPTP as catalytically inert protein phosphatases. We propose a "head-to-toe" dimerization model for RPTPgamma/zeta that is distinct from the "inhibitory wedge" model and that provides a molecular basis for inhibitory regulation. This phosphatome resource gives an expanded insight into intrafamily PTP diversity, catalytic activity, substrate recognition, and autoregulatory self-association."""
                ,
             "normalised_abstract": "Normalisedtext.."}
            ,
            {"pubmedId": "19167335"
                , "interactionType": "phosphorylation"
                , "participant1Id": "Q10076"
                , "participant2Id": "Q11099"
                ,
             "abstract": """Protein tyrosine phosphatases (PTPs) play a critical role in regulating cellular functions by selectively dephosphorylating their substrates. Here we present 22 human PTP crystal structures that, together with prior structural knowledge, enable a comprehensive analysis of the classical PTP family. Despite their largely conserved fold, surface properties of PTPs are strikingly diverse. A potential secondary substrate-binding pocket is frequently found in phosphatases, and this has implications for both substrate recognition and development of selective inhibitors. Structural comparison identified four diverse catalytic loop (WPD) conformations and suggested a mechanism for loop closure. Enzymatic assays revealed vast differences in PTP catalytic activity and identified PTPD1, PTPD2, and HDPTP as catalytically inert protein phosphatases. We propose a "head-to-toe" dimerization model for RPTPgamma/zeta that is distinct from the "inhibitory wedge" model and that provides a molecular basis for inhibitory regulation. This phosphatome resource gives an expanded insight into intrafamily PTP diversity, catalytic activity, substrate recognition, and autoregulatory self-association."""
                ,
             "normalised_abstract": "Normalisedtext.."},
            {"pubmedId": "19167335"
                , "interactionType": "phosphorylation"
                , "participant1Id": "Q11099"
                , "participant2Id": "Q11099"
                ,
             "abstract": """Protein tyrosine phosphatases (PTPs) play a critical role in regulating cellular functions by selectively dephosphorylating their substrates. Here we present 22 human PTP crystal structures that, together with prior structural knowledge, enable a comprehensive analysis of the classical PTP family. Despite their largely conserved fold, surface properties of PTPs are strikingly diverse. A potential secondary substrate-binding pocket is frequently found in phosphatases, and this has implications for both substrate recognition and development of selective inhibitors. Structural comparison identified four diverse catalytic loop (WPD) conformations and suggested a mechanism for loop closure. Enzymatic assays revealed vast differences in PTP catalytic activity and identified PTPD1, PTPD2, and HDPTP as catalytically inert protein phosphatases. We propose a "head-to-toe" dimerization model for RPTPgamma/zeta that is distinct from the "inhibitory wedge" model and that provides a molecular basis for inhibitory regulation. This phosphatome resource gives an expanded insight into intrafamily PTP diversity, catalytic activity, substrate recognition, and autoregulatory self-association."""
                ,
             "normalised_abstract": "Normalisedtext.."}
            , {"pubmedId": "25260751"
                , "interactionType": "phosphorylation"
                , "participant1Id": "Q26401"
                , "participant2Id": "Q26401"
                ,
               "abstract": "Unlike the other MAP3Ks, MEKK1 (encoded by Map3k1) contains a PHD motif. To understand the role of this motif, we have created a knockin mutant of mouse Map3k1 (Map3k1(m) (PHD)) with an inactive PHD motif. Map3k1(m) (PHD) ES cells demonstrate that the MEKK1 PHD controls p38 and JNK activation during TGF-b, EGF and microtubule disruption signalling, but does not affect MAPK responses to hyperosmotic stress. Protein microarray profiling identified the adaptor TAB1 as a PHD substrate, and TGF-b- or EGF-stimulated Map3k1(m) (PHD) ES cells exhibit defective non-canonical ubiquitination of MEKK1 and TAB1. The MEKK1 PHD binds and mediates the transfer of Lys63-linked poly-Ub, using the conjugating enzyme UBE2N, onto TAB1 to regulate TAK1 and MAPK activation by TGF-b and EGF. Both the MEKK1 PHD and TAB1 are critical for ES-cell differentiation and tumourigenesis. Map3k1(m) (PHD) (/+) mice exhibit aberrant cardiac tissue, B-cell development, testis and T-cell signalling. "
                ,
               "normalised_abstract": "Normalisedtext.."}]

        # Act
        actual = sut.parse(input)

        # Assert
        sort_func = lambda x: "{}#{}#{}#{}".format(x["pubmedId"], x["interactionType"], x["participant1Id"],
                                                   x["participant2Id"])
        self.assertEqual(expected, sorted(list(actual), key=sort_func))

    def test_load_file(self):
        # Arrange
        sut = PubtatorAnnotationsInferenceTransformer()
        input_file = os.path.join(os.path.dirname(__file__), "data_sample_annotation", "sample_1.txt")
        expected_records = 3
        mock_text_nomaliser = MagicMock()
        mock_text_nomaliser.return_value = "Normalisedtext.."
        sut.textGeneNormaliser = mock_text_nomaliser

        mock_gene_converter = MagicMock()
        mock_gene_converter.convert.side_effect = lambda x: {x: ["Q{}".format(x)]}
        sut.geneIdConverter = mock_gene_converter

        # Act
        actual = sut.load_file(input_file)

        # Assert
        self.assertEqual(expected_records, len(list(actual)))

    def test_load_directory(self):
        # Arrange
        sut = PubtatorAnnotationsInferenceTransformer()
        input_file = os.path.join(os.path.dirname(__file__), "data_sample_annotation")
        expected_parts_len = 2
        expected_total_records = 4
        mock_text_nomaliser = MagicMock()
        mock_text_nomaliser.return_value = "Normalisedtext.."
        sut.textGeneNormaliser = mock_text_nomaliser

        mock_gene_converter = MagicMock()
        mock_gene_converter.convert.side_effect = lambda x: {x: ["Q{}".format(x)]}
        sut.geneIdConverter = mock_gene_converter

        # Act
        actual = sut.load_directory(input_file)

        # Assert
        actual_list = list(actual)
        self.assertEqual(expected_parts_len, len(actual_list))
        self.assertEqual(expected_total_records, len(list(functools.reduce(operator.iconcat, actual_list, []))))

    def test_load_directory_save(self):
        # Arrange
        sut = PubtatorAnnotationsInferenceTransformer()
        input_dir = os.path.join(os.path.dirname(__file__), "data_sample_annotation")
        dest_dir = tempfile.mkdtemp()
        expected_parts_len = 2
        expected_total_records = 4
        mock_text_nomaliser = MagicMock()
        mock_text_nomaliser.return_value = "Normalisedtext.."
        sut.textGeneNormaliser = mock_text_nomaliser

        mock_gene_converter = MagicMock()
        mock_gene_converter.convert.side_effect = lambda x: {x: ["Q{}".format(x)]}
        sut.geneIdConverter = mock_gene_converter

        # Act
        actual = sut.load_directory_save(input_dir, dest_dir)

        # Assert
        self.assertEqual(expected_parts_len, len(os.listdir(dest_dir)))

        # Assert that the length of the array within the json file matches
        total_actual = 0
        for f in os.listdir(dest_dir):
            with open(os.path.join(dest_dir, f), "r") as handle:
                total_actual += len(json.load(handle))
        self.assertEqual(expected_total_records, total_actual)
