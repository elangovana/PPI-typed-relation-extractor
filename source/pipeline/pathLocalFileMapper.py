class PathLocalFileMapper:

    @staticmethod
    def get_scheme():
        return "file"

    def __call__(self, path):
        # check if path uri
        index = path.find("://")

        ## local file
        if index == -1: return path

        # Extract just the path
        scheme = path[0:index]
        if scheme.lower() == self.get_scheme():
            return path[index + 3:]

        return ""


