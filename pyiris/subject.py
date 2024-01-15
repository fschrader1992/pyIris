# -*- coding: utf-8
"""
This class contains all properties/methods related to stimuli.
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

        self.colorspaces = []
        self.data = {}

    def save_to_file(self, path=None, directory=None, filetype="yaml"):
        """
        Save object data to file.
        :param path: location of file.
        :param directory: directory, if file name should be filled automatically.
        """

        if path is not None and "." in path:
            filetype = path.split(".")[-1]

        dt = {}
        dt.update(vars(self))
        dt["uuid"] = str(self.uuid)

        if not path:
            path_var = self.short if len(self.short) > 0 else self.uuid
            path = "subject_{}.yaml".format(path_var)
        if directory:
            path = os.path.join(directory, path)
        dump_file(dt, path, filetype)

        print("Successfully saved subject data to file {}".format(path))

    def load_from_file(self, path=None, filetype="yaml"):
        """
        Load from file.
        :param path: location of file.
        """

        if "." in path:
            filetype = path.split(".")[-1]

        with open(path, "r") as f:
            if filetype == "yaml":
                d = ruamel.yaml.YAML().load(f)
            else:
                d = json.load(f)

        for a, b in d.items():
            setattr(self, a, self.__class__(b) if isinstance(b, dict) else b)

        self.uuid = uuid.UUID(str(self.uuid))

        print("Successfully loaded subject data from file {}".format(path))
