class InteractionTypePrefixer:
    """
    Prefixes the interaction type to a column..
    Given a row  , col_to_transform is 0 and prefixer_col_index is 3:
        ["This is sample entity1 entity1", "entity1", "entity2", "phosphorylation"]
    :returns
        ['QUERYphosphorylation This is sample entity1 entity1', 'entity1', 'entity2', 'phosphorylation']
    """

    def __init__(self, col_to_transform: int, prefixer_col_index: int):
        """
        Constructor
        :param col_to_transform: The col index to transform
        :param prefixer_col_index: The col index containing the prefix
        """
        self.col_to_transform = col_to_transform
        self.prefixer_col_index = prefixer_col_index

    def __call__(self, row_x):
        row_x[self.col_to_transform] = "QUERY{} {}".format(row_x[self.prefixer_col_index], row_x[self.col_to_transform])
