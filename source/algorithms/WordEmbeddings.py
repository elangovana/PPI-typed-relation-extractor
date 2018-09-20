import gensim

class WordEmbeddings:
    def __init__(self, model_path):
        self.model_path = model_path

    def run(self):
        model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)
        w1 = "klk3"
        print(model.wv[w1])
        print(model.wv.most_similar(positive=w1))


