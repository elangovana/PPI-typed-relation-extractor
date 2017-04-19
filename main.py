import getopt
import sys

import re
from bioservices.kegg import KEGG


def extract_protein_interaction(kegg_pathway_id):
    kegg = KEGG()
    kgml_parser = kegg.parse_kgml_pathway(kegg_pathway_id)
    protein_relations = list(filter(lambda d: d['link'] in ['PPrel'], kgml_parser['relations']))
    kegg_entries = kgml_parser['entries']
    print(kegg_entries)

    for rel in protein_relations:


        ko_dnumber=list(filter(lambda d: d['id'] in [rel['entry2']], kegg_entries))[0]['name']
        ko_snumber=list(filter(lambda d: d['id'] in [rel['entry1']], kegg_entries))[0]['name']
        #gene_names
        hsa_snumber="+".join(re.findall( r"(?:\t)(.+)", kegg.link('hsa', ko_snumber)))
        hsa_dnumber="+".join(re.findall( r"(?:\t)(.+)", kegg.link('hsa', ko_dnumber)))

        uniprot_snumber= kegg.conv("uniprot", hsa_snumber)
        uniprot_dnumber= kegg.conv("uniprot", hsa_dnumber)
        for s, sv in uniprot_snumber.items():
            for d, dv in uniprot_dnumber.items():
                print(sv+ "\t" + rel['name'] + '\t' + dv)










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
