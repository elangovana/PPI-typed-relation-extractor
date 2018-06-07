[![Build Status](https://travis-ci.org/elangovana/kegg-pathway-extractor.svg?branch=master)](https://travis-ci.org/elangovana/kegg-pathway-extractor)

# kegg-pathway-extractor
Given a Kegg Pathway Id, e.g path:ko05215, extracts protein interactions defined in that pathway and the type of interaction.

# Prerequisite
1. Data
   Get the MIPS Interaction XML file & extract
   ```shell
   wget http://mips.helmholtz-muenchen.de/proj/ppi/data/mppi.gz
   gunzip mppi.gz 
   ```  

# Run Docker
docker run -i -t lanax/keggproteininteractionsextractor -v /localdir/input:/opt/data/input /localdir/output:/opt/data/output <konumber> /opt/data/input/<input_data_mips_api>  /opt/data/output 