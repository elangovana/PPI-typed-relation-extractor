FROM python:3

RUN mkdir -p /opt/program/kegg_protein_extractor
RUN yum install  python-lxml

ADD .  /opt/program/kegg_protein_extractor
RUN pip install -r /opt/program/kegg_protein_extractor/requirements.txt -t /opt/program/kegg_protein_extractor/
RUN pip install  awscli --upgrade
CMD [ "python", "/opt/program/kegg_protein_extractor/main.py" ]
