import getopt
import sys
from bioservices.kegg import KEGG


def extract_protein_interaction(kegg_pathway_id):
    kegg = KEGG()
    kgml_parser = kegg.parse_kgml_pathway(kegg_pathway_id)
    protein_relations = list(filter(lambda d: d['link'] in ['PPrel'], kgml_parser['relations']))
    print protein_relations


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
