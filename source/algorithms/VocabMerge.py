class VocabMerger:

    def __call__(self, vocab_dict_1, vocab_dict_2):
        result = vocab_dict_1.copy()

        for k in vocab_dict_2:
            if k not in result:
                result[k] = len(result)

        return result
