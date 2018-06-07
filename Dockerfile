FROM python:3

RUN mkdir /opt/program/kegg_protein_extractor
ADD .  /opt/program/kegg_protein_extractor
RUN pip install -r /opt/program/kegg_protein_extractor/requirements.txt -t /opt/program/kegg_protein_extractor/
CMD [ "python", "/opt/program/kegg_protein_extractor//main.py" ]