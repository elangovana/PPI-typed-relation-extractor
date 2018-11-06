from io import StringIO
from unittest import TestCase

from dataformatters.gnormplusPubtatorReader import GnormplusPubtatorReader


class TestGnormplusPubtatorReader(TestCase):
    def test_call(self):
        s = """19167335|a|Protein tyrosine phosphatases (PTPs) play a critical role in regulating cellular functions by selectively dephosphorylating their substrates. Here we present 22 human PTP crystal structures that, together with prior structural knowledge, enable a comprehensive analysis of the classical PTP family. Despite their largely conserved fold, surface properties of PTPs are strikingly diverse. A potential secondary substrate-binding pocket is frequently found in phosphatases, and this has implications for both substrate recognition and development of selective inhibitors. Structural comparison identified four diverse catalytic loop (WPD) conformations and suggested a mechanism for loop closure. Enzymatic assays revealed vast differences in PTP catalytic activity and identified PTPD1, PTPD2, and HDPTP as catalytically inert protein phosphatases. We propose a "head-to-toe" dimerization model for RPTPgamma/zeta that is distinct from the "inhibitory wedge" model and that provides a molecular basis for inhibitory regulation. This phosphatome resource gives an expanded insight into intrafamily PTP diversity, catalytic activity, substrate recognition, and autoregulatory self-association.
19167335	167	170	PTP	Gene	10076
19167335	287	290	PTP	Gene	10076
19167335	779	784	PTPD1	Gene	11099
19167335	797	802	HDPTP	Gene	25930
19167335	898	907	RPTPgamma	Gene	5793
19167335	741	744	PTP	Gene	10076
19167335	161	166	human	Species	9606
19167335	1096	1099	PTP	Gene	10076

25260751|a|Unlike the other MAP3Ks, MEKK1 (encoded by Map3k1) contains a PHD motif. To understand the role of this motif, we have created a knockin mutant of mouse Map3k1 (Map3k1(m) (PHD)) with an inactive PHD motif. Map3k1(m) (PHD) ES cells demonstrate that the MEKK1 PHD controls p38 and JNK activation during TGF-b, EGF and microtubule disruption signalling, but does not affect MAPK responses to hyperosmotic stress. Protein microarray profiling identified the adaptor TAB1 as a PHD substrate, and TGF-b- or EGF-stimulated Map3k1(m) (PHD) ES cells exhibit defective non-canonical ubiquitination of MEKK1 and TAB1. The MEKK1 PHD binds and mediates the transfer of Lys63-linked poly-Ub, using the conjugating enzyme UBE2N, onto TAB1 to regulate TAK1 and MAPK activation by TGF-b and EGF. Both the MEKK1 PHD and TAB1 are critical for ES-cell differentiation and tumourigenesis. Map3k1(m) (PHD) (/+) mice exhibit aberrant cardiac tissue, B-cell development, testis and T-cell signalling. 
25260751	25	30	MEKK1	Gene	26401
25260751	43	49	Map3k1	Gene	26401
25260751	153	159	Map3k1	Gene	26401
25260751	161	167	Map3k1	Gene	26401
25260751	206	212	Map3k1	Gene	26401
25260751	252	257	MEKK1	Gene	26401
25260751	271	274	p38	Gene	26416
25260751	279	282	JNK	Gene	26419
25260751	301	306	TGF-b	Gene	21803
25260751	491	496	TGF-b	Gene	21803
25260751	516	522	Map3k1	Gene	26401
25260751	591	596	MEKK1	Gene	26401
25260751	611	616	MEKK1	Gene	26401
25260751	707	712	UBE2N	Gene	93765
25260751	736	740	TAK1	Gene	26409
25260751	764	769	TGF-b	Gene	21803
25260751	788	793	MEKK1	Gene	26401
25260751	868	874	Map3k1	Gene	26401
25260751	147	152	mouse	Species	10090
25260751	889	893	mice	Species	10090

"""

        expected = [{'id': '19167335', 'type': 'a',
                     'text': 'Protein tyrosine phosphatases (PTPs) play a critical role in regulating cellular functions by selectively dephosphorylating their substrates. Here we present 22 human PTP crystal structures that, together with prior structural knowledge, enable a comprehensive analysis of the classical PTP family. Despite their largely conserved fold, surface properties of PTPs are strikingly diverse. A potential secondary substrate-binding pocket is frequently found in phosphatases, and this has implications for both substrate recognition and development of selective inhibitors. Structural comparison identified four diverse catalytic loop (WPD) conformations and suggested a mechanism for loop closure. Enzymatic assays revealed vast differences in PTP catalytic activity and identified PTPD1, PTPD2, and HDPTP as catalytically inert protein phosphatases. We propose a "head-to-toe" dimerization model for RPTPgamma/zeta that is distinct from the "inhibitory wedge" model and that provides a molecular basis for inhibitory regulation. This phosphatome resource gives an expanded insight into intrafamily PTP diversity, catalytic activity, substrate recognition, and autoregulatory self-association.',
                     'annotations': [
                         {'start': '167', 'end': '170', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'},
                         {'start': '287', 'end': '290', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'},
                         {'start': '779', 'end': '784', 'name': 'PTPD1', 'type': 'Gene', 'normalised_id': '11099'},
                         {'start': '797', 'end': '802', 'name': 'HDPTP', 'type': 'Gene', 'normalised_id': '25930'},
                         {'start': '898', 'end': '907', 'name': 'RPTPgamma', 'type': 'Gene', 'normalised_id': '5793'},
                         {'start': '741', 'end': '744', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'},
                         {'start': '161', 'end': '166', 'name': 'human', 'type': 'Species', 'normalised_id': '9606'},
                         {'start': '1096', 'end': '1099', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'}]}
            , {'id': '25260751', 'type': 'a',
               'text': 'Unlike the other MAP3Ks, MEKK1 (encoded by Map3k1) contains a PHD motif. To understand the role of this motif, we have created a knockin mutant of mouse Map3k1 (Map3k1(m) (PHD)) with an inactive PHD motif. Map3k1(m) (PHD) ES cells demonstrate that the MEKK1 PHD controls p38 and JNK activation during TGF-b, EGF and microtubule disruption signalling, but does not affect MAPK responses to hyperosmotic stress. Protein microarray profiling identified the adaptor TAB1 as a PHD substrate, and TGF-b- or EGF-stimulated Map3k1(m) (PHD) ES cells exhibit defective non-canonical ubiquitination of MEKK1 and TAB1. The MEKK1 PHD binds and mediates the transfer of Lys63-linked poly-Ub, using the conjugating enzyme UBE2N, onto TAB1 to regulate TAK1 and MAPK activation by TGF-b and EGF. Both the MEKK1 PHD and TAB1 are critical for ES-cell differentiation and tumourigenesis. Map3k1(m) (PHD) (/+) mice exhibit aberrant cardiac tissue, B-cell development, testis and T-cell signalling. ',
               'annotations': [{'start': '25', 'end': '30', 'name': 'MEKK1', 'type': 'Gene', 'normalised_id': '26401'},
                               {'start': '43', 'end': '49', 'name': 'Map3k1', 'type': 'Gene', 'normalised_id': '26401'},
                               {'start': '153', 'end': '159', 'name': 'Map3k1', 'type': 'Gene',
                                'normalised_id': '26401'},
                               {'start': '161', 'end': '167', 'name': 'Map3k1', 'type': 'Gene',
                                'normalised_id': '26401'},
                               {'start': '206', 'end': '212', 'name': 'Map3k1', 'type': 'Gene',
                                'normalised_id': '26401'},
                               {'start': '252', 'end': '257', 'name': 'MEKK1', 'type': 'Gene',
                                'normalised_id': '26401'},
                               {'start': '271', 'end': '274', 'name': 'p38', 'type': 'Gene', 'normalised_id': '26416'},
                               {'start': '279', 'end': '282', 'name': 'JNK', 'type': 'Gene', 'normalised_id': '26419'},
                               {'start': '301', 'end': '306', 'name': 'TGF-b', 'type': 'Gene',
                                'normalised_id': '21803'},
                               {'start': '491', 'end': '496', 'name': 'TGF-b', 'type': 'Gene',
                                'normalised_id': '21803'},
                               {'start': '516', 'end': '522', 'name': 'Map3k1', 'type': 'Gene',
                                'normalised_id': '26401'},
                               {'start': '591', 'end': '596', 'name': 'MEKK1', 'type': 'Gene',
                                'normalised_id': '26401'},
                               {'start': '611', 'end': '616', 'name': 'MEKK1', 'type': 'Gene',
                                'normalised_id': '26401'},
                               {'start': '707', 'end': '712', 'name': 'UBE2N', 'type': 'Gene',
                                'normalised_id': '93765'},
                               {'start': '736', 'end': '740', 'name': 'TAK1', 'type': 'Gene', 'normalised_id': '26409'},
                               {'start': '764', 'end': '769', 'name': 'TGF-b', 'type': 'Gene',
                                'normalised_id': '21803'},
                               {'start': '788', 'end': '793', 'name': 'MEKK1', 'type': 'Gene',
                                'normalised_id': '26401'},
                               {'start': '868', 'end': '874', 'name': 'Map3k1', 'type': 'Gene',
                                'normalised_id': '26401'},
                               {'start': '147', 'end': '152', 'name': 'mouse', 'type': 'Species',
                                'normalised_id': '10090'},
                               {'start': '889', 'end': '893', 'name': 'mice', 'type': 'Species',
                                'normalised_id': '10090'}]}

                    ]

        handle = StringIO(s)
        sut = GnormplusPubtatorReader()

        # Act
        actual = sut(handle)

        # Assert
        self.assertSequenceEqual(list(actual), expected)
