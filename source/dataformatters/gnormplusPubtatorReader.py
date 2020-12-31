import logging

"""
Reads a Pubtator formatted context and returns json

e.g. input:

19167335|a|Protein tyrosine phosphatases (PTPs) play a critical role in regulating cellular functions by selectively dephosphorylating their substrates. Here we present 22 human PTP crystal structures that, together with prior structural knowledge, enable a comprehensive analysis of the classical PTP family. Despite their largely conserved fold, surface properties of PTPs are strikingly diverse. A potential secondary substrate-binding pocket is frequently found in phosphatases, and this has implications for both substrate recognition and development of selective inhibitors. Structural comparison identified four diverse catalytic loop (WPD) conformations and suggested a mechanism for loop closure. Enzymatic assays revealed vast differences in PTP catalytic activity and identified PTPD1, PTPD2, and HDPTP as catalytically inert protein phosphatases. We propose a "head-to-toe" dimerization model for RPTPgamma/zeta that is distinct from the "inhibitory wedge" model and that provides a molecular basis for inhibitory regulation. This phosphatome resource gives an expanded insight into intrafamily PTP diversity, catalytic activity, substrate recognition, and autoregulatory self-association.
19167335	167	170	PTP	Gene	10076
19167335	287	290	PTP	Gene	10076
19167335	779	784	PTPD1	Gene	11099


25260751|a|Unlike the other MAP3Ks, MEKK1 (encoded by Map3k1) contains a PHD motif. To understand the role of this motif, we have created a knockin mutant of mouse Map3k1 (Map3k1(m) (PHD)) with an inactive PHD motif. Map3k1(m) (PHD) ES cells demonstrate that the MEKK1 PHD controls p38 and JNK activation during TGF-b, EGF and microtubule disruption signalling, but does not affect MAPK responses to hyperosmotic stress. Protein microarray profiling identified the adaptor TAB1 as a PHD substrate, and TGF-b- or EGF-stimulated Map3k1(m) (PHD) ES cells exhibit defective non-canonical ubiquitination of MEKK1 and TAB1. The MEKK1 PHD binds and mediates the transfer of Lys63-linked poly-Ub, using the conjugating enzyme UBE2N, onto TAB1 to regulate TAK1 and MAPK activation by TGF-b and EGF. Both the MEKK1 PHD and TAB1 are critical for ES-cell differentiation and tumourigenesis. Map3k1(m) (PHD) (/+) mice exhibit aberrant cardiac tissue, B-cell development, testis and T-cell signalling. 
25260751	25	30	MEKK1	Gene	26401
25260751	43	49	Map3k1	Gene	26401
25260751	153	159	Map3k1	Gene	26401
25260751	271	274	p38	Gene	26416
25260751	279	282	JNK	Gene	26419


Output:
[{'id': '19167335', 
  'type': 'a',
  'text': 'Protein tyrosine phosphatases (PTPs) play a critical role in regulating cellular functions by selectively dephosphorylating their substrates. Here we present 22 human PTP crystal structures that, together with prior structural knowledge, enable a comprehensive analysis of the classical PTP family. Despite their largely conserved fold, surface properties of PTPs are strikingly diverse. A potential secondary substrate-binding pocket is frequently found in phosphatases, and this has implications for both substrate recognition and development of selective inhibitors. Structural comparison identified four diverse catalytic loop (WPD) conformations and suggested a mechanism for loop closure. Enzymatic assays revealed vast differences in PTP catalytic activity and identified PTPD1, PTPD2, and HDPTP as catalytically inert protein phosphatases. We propose a "head-to-toe" dimerization model for RPTPgamma/zeta that is distinct from the "inhibitory wedge" model and that provides a molecular basis for inhibitory regulation. This phosphatome resource gives an expanded insight into intrafamily PTP diversity, catalytic activity, substrate recognition, and autoregulatory self-association.',
  'annotations': [
         {'start': '167', 'end': '170', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'},
         {'start': '287', 'end': '290', 'name': 'PTP', 'type': 'Gene', 'normalised_id': '10076'},
         {'start': '779', 'end': '784', 'name': 'PTPD1', 'type': 'Gene', 'normalised_id': '11099'}
    ]}
,{'id': '25260751',
  'type': 'a',
  'text': 'Unlike the other MAP3Ks, MEKK1 (encoded by Map3k1) contains a PHD motif. To understand the role of this motif, we have created a knockin mutant of mouse Map3k1 (Map3k1(m) (PHD)) with an inactive PHD motif. Map3k1(m) (PHD) ES cells demonstrate that the MEKK1 PHD controls p38 and JNK activation during TGF-b, EGF and microtubule disruption signalling, but does not affect MAPK responses to hyperosmotic stress. Protein microarray profiling identified the adaptor TAB1 as a PHD substrate, and TGF-b- or EGF-stimulated Map3k1(m) (PHD) ES cells exhibit defective non-canonical ubiquitination of MEKK1 and TAB1. The MEKK1 PHD binds and mediates the transfer of Lys63-linked poly-Ub, using the conjugating enzyme UBE2N, onto TAB1 to regulate TAK1 and MAPK activation by TGF-b and EGF. Both the MEKK1 PHD and TAB1 are critical for ES-cell differentiation and tumourigenesis. Map3k1(m) (PHD) (/+) mice exhibit aberrant cardiac tissue, B-cell development, testis and T-cell signalling. ',
  'annotations': [
        {'start': '25', 'end': '30', 'name': 'MEKK1', 'type': 'Gene', 'normalised_id': '26401'},
        {'start': '43', 'end': '49', 'name': 'Map3k1', 'type': 'Gene', 'normalised_id': '26401'},
        {'start': '153', 'end': '159', 'name': 'Map3k1', 'type': 'Gene', 'normalised_id': '26401'},
        {'start': '271', 'end': '274', 'name': 'p38', 'type': 'Gene', 'normalised_id': '26416'},
        {'start': '279', 'end': '282', 'name': 'JNK', 'type': 'Gene', 'normalised_id': '26419'}
        ]}

]

"""


class GnormplusPubtatorReader:

    def __init__(self):
        pass

    def __call__(self, handle) -> iter:

        for header in handle:
            # Skip blank lines
            if header.strip("\n\t\s") == "": continue

            record = {}

            header_parts = header.split("|")
            record["id"] = header_parts[0]
            record["type"] = header_parts[1]
            record["text"] = header_parts[2].strip("\n")
            annotations = []
            # Loop through annotation
            while True:
                try:
                    annotation_txt = next(handle)
                except StopIteration:
                    return

                if annotation_txt == "\n": break
                # Expected annotation format
                # 19167335        167     170     PTP     Gene    10076
                annotation_parts = annotation_txt.split("\t")
                start_pos, end_pos = annotation_parts[1], annotation_parts[2]
                name = annotation_parts[3]
                type = annotation_parts[4]
                normalised_id = annotation_parts[5].strip("\n")

                annotations.append(
                    {"start": start_pos, "end": end_pos, "name": name, "type": type, "normalised_id": normalised_id})

            record["annotations"] = annotations
            yield record

    @property
    def logger(self):
        return logging.getLogger(__name__)
