import importlib
import os
import pkgutil

from modelnetworks.NetworkFactoryBase import NetworkFactoryBase


class NetworkFactoryLocator:
    """
    General network factory that automatically loads network factories that are subclasses of NetworkFactoryBase
    """

    def __init__(self):
        # Expect the datset factory is under datasets path under the parent of the __file__
        datasets_base_dir = "modelnetworks"
        base_class = NetworkFactoryBase

        # search path
        search_path = os.path.join(os.path.dirname(__file__), "..", datasets_base_dir)

        # load subclasses of CustomDatasetFactoryBase from datasets
        for _, name, _ in pkgutil.iter_modules([search_path]):
            importlib.import_module(datasets_base_dir + '.' + name)

        self._class_name_class_dict = {cls.__name__: cls for cls in base_class.__subclasses__()}

    @property
    def factory_names(self):
        """
        Returns the names of subclasses of NetworkFactoryBase that can be dynamically loaded
        :return:
        """
        return list(self._class_name_class_dict.keys())

    def get_factory(self, class_name):
        """
        Returns a dataset factory object
        :param class_name: The name of the NetworkFactoryBase  class, see property dataset_factory_names to obtain valid list of class names
        :return:
        """
        if class_name in self._class_name_class_dict:
            return self._class_name_class_dict[class_name]()
        else:
            raise ModuleNotFoundError("Module should be in {}".format(self.factory_names))
