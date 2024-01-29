# -*- coding: utf-8
"""
This class contains all properties/methods related to stimuli.
Latest version: 2.0.0.
"""

import os
import uuid
import json
import ruamel.yaml

from .functions import dump_file


class Subject:
    """
    Class Subject
    """

    def __init__(self, short="", name="", surname="", birthday="", notes=""):
        self.uuid = uuid.uuid4()
        self.short = short
        self.name = name
        self.surname = surname
        self.birthday = birthday
        self.notes = notes

    def save_to_file(self, path=None, directory=None, filetype="yaml"):
        """
        Save subject data to file.
        :param path: Location of file. Default in None.
        :param directory: Directory, if file name should be filled automatically.
        :param filetype: Filetype, "json" or "yaml".
               Default is "yaml" but set to file extension if found in path.
        :return: True.
        """
        if path is not None and "." in path:
            filetype = path.split(".")[-1]

        dt = {}
        dt.update(vars(self))
        dt["uuid"] = str(self.uuid)

        if not path:
            path_var = self.short if len(self.short) > 0 else self.uuid
            path = "subject_{}.{}".format(path_var, filetype)
        if directory:
            path = os.path.join(directory, path)
        dump_file(dt, path, filetype)

        print("Successfully saved subject data to file {}".format(path))

        return True

    def load_from_file(self, path=None, filetype="yaml"):
        """
        Load subject from file.
        :param path: Location of file. Default in None.
        :param filetype: Filetype, "json" or "yaml".
               Default is "yaml" but set to file extension if found in path.
        :return: True.
        """

        if "." in path:
            filetype = path.split(".")[-1]
        else:
            path += "." + filetype

        with open(path, "r") as f:
            if filetype == "yaml":
                d = ruamel.yaml.YAML().load(f)
            else:
                d = json.load(f)
        if "colorspaces" in d.keys():
            del d["colorspaces"]
        if "data" in d.keys():
            del d["data"]

        for a, b in d.items():
            setattr(self, a, self.__class__(b) if isinstance(b, dict) else b)

        self.uuid = uuid.UUID(str(self.uuid))

        print("Successfully loaded subject data from file {}".format(path))

        return True
