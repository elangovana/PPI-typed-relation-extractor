import getopt
import sys

import re
from bioservices.kegg import KEGG


def extract_protein_interaction(kegg_pathway_id):
    kegg = KEGG()
    kgml_parser = kegg.parse_kgml_pathway(kegg_pathway_id)
    protein_relations = list(filter(lambda d: d['link'] in ['PPrel'], kgml_parser['relations']))
    kegg_entries = kgml_parser['entries']
    for rel in protein_relations:
        uniprot_dnumber = get_uniprot_numbers(kegg, kegg_entries, rel['entry2'])
        uniprot_snumber = get_uniprot_numbers(kegg, kegg_entries, rel['entry1'])

        for s, sv in uniprot_snumber.items():
            for d, dv in uniprot_dnumber.items():
                print(sv+ "\t" + rel['name'] + '\t' + dv+ '\t' )


def get_uniprot_numbers(kegg, kegg_entries, entry_id):
    regex_hsa = r"(?:\t)(.+)"
    uniprot_number={}
    ko_number = list(filter(lambda d: d['id'] in [entry_id], kegg_entries))[0]['name']
    ko_number_map = kegg.link('hsa', ko_number)


    hsa_number_list = re.findall(regex_hsa, str(ko_number_map))
    if  len(hsa_number_list) > 0:
        hsa_number = "+".join(hsa_number_list)
        uniprot_number = kegg.conv("uniprot", hsa_number)


    return uniprot_number


def main(argv):
    # default kegg pathway id for sample test run
    keg_pathway_id = "path:ko05215"
    try:
        opts, args = getopt.getopt(argv, "hp", ["pathwayid="])
    except getopt.GetoptError:
        print 'main.py -p <kegg_pathway_id>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print '-p <kegg_pathway_id>'
            print 'Eg:'
            print '-p path:ko05215'
            sys.exit()
        elif opt in ("-p", "--pathwayid"):
            keg_pathway_id = int(arg)
    extract_protein_interaction(keg_pathway_id)


if __name__ == "__main__":
    main(sys.argv[1:])
