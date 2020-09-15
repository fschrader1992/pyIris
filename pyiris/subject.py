# -*- coding: utf-8
"""
This class contains all properties/methods related to stimuli.
"""

import os
import uuid
import json
import codecs
import argparse


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

    def save_to_file(self, path=None, directory=None):
        """
        Save object data to file.
        :param path: location of file.
        :param directory: directory, if file name should be filled automatically.
        """
        dt = {}
        dt.update(vars(self))
        dt["uuid"] = str(self.uuid)

        if not path:
            path_var = self.short if len(self.short) > 0 else self.id
            path = "{}.subject".format(path_var)
        if directory:
            path = os.path.join(directory, path)
        json.dump(dt, codecs.open(path, 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True, indent=4)

        print("Successfully saved subject data to file {}".format(path))

    def load_from_file(self, path=None):
        """
        Load from file.
        :param path: location of file.
        """

        with open(path, "r") as f:
            d = json.load(f)
        for a, b in d.items():
            setattr(self, a, self.__class__(b) if isinstance(b, dict) else b)

        self.uuid = uuid.UUID(str(self.uuid))

        print("Successfully loaded subject data from file {}".format(path))
