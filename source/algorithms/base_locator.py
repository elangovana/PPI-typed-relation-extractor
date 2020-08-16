# *****************************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************

import importlib


class BaseLocator:
    """
 locator
    """

    def __init__(self, base_class):
        self._base_class = base_class

    def get(self, module_class_name):
        """
        Returns a  factory object
        :param module_class_name: The name of the class to import, e.g. base_locator.BaseLocator
        :return:
        """

        module_parts = module_class_name.split(".")
        module_name = ".".join(module_parts[0:-1])
        class_name = module_parts[-1]

        print("Attempting to load module {}, class {}".format(module_name, class_name))
        importlib.import_module(module_name)

        # Validate that the class is a subclass
        class_name_class_dict = {cls.__name__: cls for cls in self._base_class.__subclasses__()}

        if class_name not in class_name_class_dict:
            raise ValueError(
                f"Could not find load class {class_name}. Make sure that it is a subclass of {self._base_class} and the module name {module_name} is correct")

        return class_name_class_dict[class_name]()
