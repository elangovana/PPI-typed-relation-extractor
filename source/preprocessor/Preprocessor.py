class Preprocessor:


    """
    Applies a bunch of transformations
    """


    def __init__(self, preprocessors):
        self.preprocessors = preprocessors


    def __call__(self, row):
        transformed_row = row
        for p in self.preprocessors:
            transformed_row = p(transformed_row)

        return transformed_row
