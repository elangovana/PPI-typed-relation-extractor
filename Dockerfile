FROM python:3

RUN mkdir -p /opt/program/kegg_protein_extractor
#RUN apt-get install python3-lxml

COPY .  /opt/program/kegg_protein_extractor
RUN pip install -r /opt/program/kegg_protein_extractor/requirements.txt -t /opt/program/kegg_protein_extractor/
RUN pip install  awscli --upgrade
RUN export PYTHONPATH=/opt/program/kegg_protein_extractor
CMD [ "bash", "/opt/program/kegg_protein_extractor/scripts/run_pipeline.py" ]
