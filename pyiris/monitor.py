"""
Fill an object of the psychopy Monitor class with values from the monitor config file
(simply in order to keep track of only one file).
Latest version: 2.0.0.
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
        self.name = name if name else "temp"
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

        if "width_mm" in d["size"].keys():
            wmm = d["size"]["width_mm"]/10.
        elif "width_cm" in d["size"].keys():
            wmm = d["size"]["width_cm"]
        else:
            wmm = d["size"]["width"]
        self.setWidth(wmm)

        wpx = d["preferred_mode"]["width_px"] if "width_px" in d["preferred_mode"].keys()\
            else d["preferred_mode"]["width"]
        hpx = d["preferred_mode"]["height_px"] if "height_px" in d["preferred_mode"].keys()\
            else d["preferred_mode"]["height"]
        self.setSizePix([wpx, hpx])

        if "distance_mm" in d["preferred_mode"].keys():
            distance = d["preferred_mode"]["distance_mm"]/10.
        elif "distance_cm" in d["preferred_mode"].keys():
            distance = d["preferred_mode"]["distance_cm"]
        else:
            distance = d["preferred_mode"]["distance"]
        self.setDistance(distance)

        # for window settings
        self.bpc = d["preferred_mode"]["color-depth"]
        self.refresh = d["preferred_mode"]["refresh"]
