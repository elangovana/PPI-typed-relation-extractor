from unittest import TestCase
from algorithms.WordEmbeddings import WordEmbeddings

class TestEmbeddings(TestCase):
    def test_run(self):
        #Arrange
        sut = WordEmbeddings("./GoogleNews-vectors-negative300.bin")

        sut.run()
