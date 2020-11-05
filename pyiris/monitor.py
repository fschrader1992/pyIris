"""
Fill an object of the psychopy Monitor class with values from the monitor config file
(simply in order to having to keep track of only one file).
"""

import datetime
import uuid
import yaml

import psychopy.monitors as mon


class Monitor(mon.Monitor):
    """
    Monitor Class.
    """

    def __init__(self, name=None, settings_path=None):
        self.uuid = uuid.uuid4()
        self.date = datetime.datetime.now()
        self.name = name
        super().__init__(self.name)

        self.bpc = None
        self.refresh = None

        self.settings_path = settings_path
        if settings_path:
            self.add_settings()

    def add_settings(self, settings_path=None):
        """
        Add the settings from the file.
        """
        if not settings_path:
            settings_path = self.settings_path
        else:
            self.settings_path = settings_path

        # load settings file
        with open(settings_path, "r") as file:
            d = yaml.load(file, Loader=yaml.FullLoader)
        self.name = d["id"]
        self.setWidth(d["size"]["width"]/10.)
        self.setSizePix([d["preferred_mode"]["width"], d["preferred_mode"]["height"]])
        self.setDistance(d["preferred_mode"]["distance"])

        # for window settings
        self.bpc = d["preferred_mode"]["color-depth"]
        self.refresh = d["preferred_mode"]["refresh"]
