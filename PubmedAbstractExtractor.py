# coding=utf-8
import logging

import os
import tempfile

import pandas as pd
import requests
import xml.etree.cElementTree as ElementTree
from urllib.parse import urlencode

class PubmedAbstractExtractor:
    def __init__(self, pubmed_baseurl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"):
        self._logger = logging.getLogger(__name__)
        self.pubmed_baseurl = pubmed_baseurl

    def extract_abstract_by_pubmedid(self, pubmed_id_list):
        self._logger.info("Extracting pubmed abstract")

        query_string = urlencode({'db': 'pubmed'
                                  ,'id': ','.join(pubmed_id_list)
                                  ,'retmode':'abstract'
                                  ,'rettype':'xml'})

        uri = "{}?{}".format( self.pubmed_baseurl, query_string)
        self._logger.debug("Extracting pubmed abstract from url {}".format(uri))

        # Downloading pubmed abstracts Xml file
        r = requests.get(uri, allow_redirects=True)
        with tempfile.TemporaryFile(suffix=".csv", mode="w+r") as tmpfile:
            self._logger.info("Downloading {} to temp file".format(uri))
            tmpfile.write(r.content)
            tmpfile.seek(0)

            # Start Extracting abstracts
            self.extract(tmpfile)

    def extract(self, handle):
        """
        Extracts pubmed abstracts as xml
        :param pubmed xml file handle:
        """
        self._logger.info("Running extract for pubmedarticle...")
        result_arr = []
        for pubmed_article in self._iter_elements_by_name(handle, "PubmedArticle"):

            pubmed_id = pubmed_article.find("MedlineCitation/PMID").text

            article_ele = pubmed_article.find("MedlineCitation/Article")
            if article_ele is None:
                continue

            title_ele = article_ele.find("ArticleTitle")
            if title_ele is None:
                continue
            article_title = title_ele.text


            abstract_ele = article_ele.find("Abstract/AbstractText")
            if abstract_ele is None:
                continue
            article_abstract = abstract_ele.text

            self._logger.debug(
                "{}: {} \n -------\n {} \n \n".format(pubmed_id, article_title, article_abstract))

            result_arr.append({
                "id": pubmed_id
                ,"title":article_title
                ,'abstract':article_abstract

            })

        df_result = pd.DataFrame(result_arr)
        return df_result

    def _iter_elements_by_name(self, handle, name):
        events = ElementTree.iterparse(handle, events=("start", "end",))
        _, root = next(events)  # Grab the root element.
        for event, elem in events:
            if event == "end" and elem.tag == name:
                yield elem
                elem.clear()


