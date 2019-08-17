class Collator:

    def __init__(self):
        pass

    def __call__(self, batch):
        y = []
        feature_len = len(batch[0][0])
        x = [[] for _ in range(feature_len)]
        for idx, (xi, yi) in enumerate(batch):
            for fi, f in enumerate(xi):
                x[fi].append(f)
            y.append(yi)

        x = [tuple(f) for f in x]

        return [x, y]
