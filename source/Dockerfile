FROM python:3.6.5

RUN mkdir -p /opt/program/kegg_protein_extractor
#RUN apt-get install python3-lxml

#Install FTP Client
RUN apt-get update && apt-get install -y ftp

RUN mkdir -p /usr/src/ftp
WORKDIR /usr/src/ftp

#Set up kegg extractor
COPY ./  /opt/program/kegg_protein_extractor
RUN ls -la /opt/program/kegg_protein_extractor
RUN pip install -r /opt/program/kegg_protein_extractor/requirements_prod.txt
RUN pip install  awscli --upgrade

#Set PythonPath
ENV PYTHONPATH /opt/program/kegg_protein_extractor

WORKDIR /opt/program/kegg_protein_extractor
#Example run
CMD ["bash", "scripts/run_pipeline_download_esindex.sh", "/data" , "human_01*.xml"]
