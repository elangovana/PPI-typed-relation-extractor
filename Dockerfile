FROM python:3

RUN mkdir -p /opt/program/kegg_protein_extractor
#RUN apt-get install python3-lxml

#Install FTP Client
RUN apt-get update && apt-get install -y ftp

RUN mkdir -p /usr/src/ftp
WORKDIR /usr/src/ftp

#Set up kegg extractor
COPY ./  /opt/program/kegg_protein_extractor
RUN ls -la /opt/program/kegg_protein_extractor
RUN pip install -r /opt/program/kegg_protein_extractor/requirements.txt -t /opt/program/kegg_protein_extractor/
RUN pip install  awscli --upgrade

WORKDIR /opt/program/kegg_protein_extractor
CMD  ["bash", "/opt/program/kegg_protein_extractor/scripts/run_pipeline.sh"]
