from sklearn.pipeline import Pipeline


class LabelPipeline:

    def __init__(self, label_reshaper, label_encoder):
        self._label_encoder = label_encoder
        self._label_reshaper = label_reshaper
        self._label_pipeline = None

    @property
    def label_reverse_encoder_func(self):
        return self._label_encoder.inverse_transform

    def transform(self, data_loader):
        # Unbatch Y
        return self._label_pipeline.transform(data_loader)

    def fit_transform(self, data_loader):
        self.fit(data_loader)
        return self.transform(data_loader)

    def fit(self, data_loader):
        self._label_pipeline = Pipeline(steps=
                                        [("label_encoder", self._label_encoder)
                                            , ("label_reshaper", self._label_reshaper)])

        for name, p in self._label_pipeline.steps:
            p.fit(data_loader)
