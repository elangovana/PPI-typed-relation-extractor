import functools
import logging
import tempfile
from urllib.parse import urlencode
from xml.etree import cElementTree as ElementTree

import requests


class PubmedAbstractExtractor:
    def __init__(self, pubmed_baseurl="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"):
        self._logger = logging.getLogger(__name__)
        self.pubmed_baseurl = pubmed_baseurl

    def extract_abstract_by_pubmedid(self, pubmed_id_list):
        # One item
        if type(pubmed_id_list) is not list:
            return self._retrieve_single_abstracts(pubmed_id_list)

        # Single length list
        if len(pubmed_id_list) == 1:
            return self._retrieve_single_abstracts(pubmed_id_list[0])

        return self._retrieve_abstracts(pubmed_id_list)

    @functools.lru_cache(maxsize=128)
    def _retrieve_single_abstracts(self, pubmedid):
        return self._retrieve_abstracts([pubmedid])

    def _retrieve_abstracts(self, pubmed_id_list):
        self._logger.info("Extracting pubmed abstract {}".format(",".join(pubmed_id_list)))
        query_string = urlencode({'db': 'pubmed'
                                     , 'id': ','.join(pubmed_id_list)
                                     , 'retmode': 'abstract'
                                     , 'rettype': 'xml'})
        uri = "{}?{}".format(self.pubmed_baseurl, query_string)
        self._logger.debug("Extracting pubmed abstract from url {}".format(uri))
        # Downloading pubmed abstracts Xml file
        r = requests.get(uri, allow_redirects=True)
        with tempfile.TemporaryFile(suffix=".csv", mode="wb+") as tmphandle:
            self._logger.debug("Downloading {} to temp file".format(uri))
            tmphandle.write(r.content)
            tmphandle.flush()
            tmphandle.seek(0)

            # Start Extracting abstracts
            return self.extract(tmphandle)

    def extract(self, handle):
        """
        Extracts pubmed abstracts as xml
        :param pubmed xml file handle:
        """
        self._logger.debug("Running extract for pubmedarticle...")
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
                , "title": article_title
                , 'abstract': article_abstract

            })

        return result_arr

    def _iter_elements_by_name(self, handle, name):
        try:
            events = ElementTree.iterparse(handle, events=("start", "end",))
            _, root = next(events)  # Grab the root element.
        except Exception as e:
            self._logger.warning("{}".format(e))
            handle.seek(0)
            msg = handle.read().decode("utf-8")
            self._logger.warning("Could not parse XML format {}".format(msg))

            raise e

        for event, elem in events:
            if event == "end" and elem.tag == name:
                yield elem
                elem.clear()
