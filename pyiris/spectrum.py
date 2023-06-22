# -*- coding: utf-8
"""
This class contains all properties/methods related to spectra.
"""

import os
import datetime
import uuid
import h5py
import numpy as np
import nixio as nix

from pathlib import Path
from psychopy import logging, visual, event

from .pr655 import PR655
from .monitor import Monitor


class Spectrum:
    """
    Class Spectrum
    """

    def __init__(self, photometer=None, colors=None, stepsize=None, monitor_settings_path=None):
        self.uuid = uuid.uuid4()
        self.date = datetime.datetime.now()
        self.photometer = photometer
        self.colors = colors
        self.stepsize = stepsize

        self.monitor_settings_path = monitor_settings_path
        self.monitor = None

        self.names = []
        self.spectra = {}

        # print errors
        logging.console.setLevel(logging.ERROR)

        if colors:
            self.names = colors
        if stepsize:
            self.create_colorlist()
        if monitor_settings_path:
            self.add_monitor_settings(monitor_settings_path)

    def add_monitor_settings(self, monitor_settings_path=None):
        """
        Get the monitor settings.
        :param monitor_settings_path: Path to configuration file.
        """

        if monitor_settings_path is None:
            monitor_settings_path = self.monitor_settings_path

        self.monitor = Monitor(settings_path=monitor_settings_path)

    def create_colorlist(self, stepsize=None):
        """
        Set list of rgb-color codes to be used for spectra.
        :param stepsize: Difference between two color steps (between 0. and 1.)
        """
        if stepsize:
            self.stepsize = stepsize
        self.colors = []
        for step in np.arange(0. + self.stepsize, 1. + self.stepsize, self.stepsize):
            self.colors += [np.asarray([step, 0., 0.])]
            self.colors += [np.asarray([0., step, 0.])]
            self.colors += [np.asarray([0., 0., step])]
            self.colors += [np.asarray([step, step, step])]

    def add_pr655(self, port="/dev/ttyUSB0"):
        """
        Automatically add PR655, if connected.
        """
        self.photometer = PR655(port=port)

    def add_spectrum(self, name):
        """
        Measure and save spectrum from photometer.
        :param name: Name this measurement has. With measure_colors thi is equal to rgb-color code.
        """
        self.spectra[name, "luminance"] = self.photometer.getLum()
        nm, power = self.photometer.getLastSpectrum(parse=True)
        self.spectra[name, "wavelength"] = nm
        self.spectra[name, "power"] = power
        self.names += [name]

        # get other data
        self.spectra[name, "tristim"] = self.photometer.getLastTristim()
        self.spectra[name, "uv"] = self.photometer.getLastUV()
        self.spectra[name, "xy"] = self.photometer.getLastXY()
        self.spectra[name, "colortemp"] = self.photometer.getLastColorTemp()

    def add_mock_spectrum(self, name, i):
        """
        TODO: REMOVE
        :param name: Name of patch.
        """
        mon_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     '../example/spectra-20150825T1554.h5')
        mon_spectra = h5py.File(mon_file_path, 'r')

        lam_start_ms = 380.
        lam_fin_ms = lam_start_ms + 4. * len(mon_spectra['spectra'][0])
        lams_ms = np.arange(lam_start_ms, lam_fin_ms, 4)

        self.spectra[name, "luminance"] = mon_spectra["luminance"][i]
        self.spectra[name, "wavelength"] = lams_ms
        self.spectra[name, "power"] = mon_spectra["spectra"][i]
        self.names += [name]

    def measure_colors(self, win_h=1200, win_w=1800):
        """
        Measure the spectra for each color stimulus.
        """

        win = visual.Window([win_h, win_w], fullscr=True)
        if self.monitor:
            win.monitor = self.monitor

        for color in self.colors:
            # draw stimulus
            # get psychopy color range
            show_color = 2. * color - 1.
            rect = visual.Rect(win=win, width=win_w, height=win_h)
            rect.fillColorSpace = "rgb"
            rect.fillColor = show_color
            rect.lineColorSpace = "rgb"
            rect.lineColor = show_color
            rect.draw()
            win.flip()
            # measure spectrum
            self.add_spectrum(name=str(color))

        win.close()

        # set date of last measurement
        self.date = datetime.datetime.now()

    def measure_patch_colors(self, win_h=1200, win_w=1800):
        """
        Measure the spectra for each color stimulus, for stimuli
        in different areas of the screen.
        """

        # shorter test-color list, around luminance values used in experiments
        self.colors = []
        for step in np.arange(0.55, 0.8, 0.05):
            self.colors += [np.asarray([step, 0., 0.])]
            self.colors += [np.asarray([0., step, 0.])]
            self.colors += [np.asarray([0., 0., step])]
            self.colors += [np.asarray([step, step, step])]

        win = visual.Window([win_h, win_w], fullscr=True)
        if self.monitor:
            win.monitor = self.monitor


        info_msg = visual.TextStim(self.win, '', color='black',pos=(0, 10), height=0.75)
        # iterate through 4 dot positions and repeat each measurement 6 times
        xys = [[-1.5, 1.5], [1.5, 1.5], [-1.5, -1.5], [1.5, -1.5]]
        xy_labels = ['up_left', 'up_right', 'down_left', 'down_right']
        for xy_label, xy in zip(xy_labels, xys):
            # start with stimulus in order to adjust photometer
            info_msg.text = 'Please adjust the photometer to the stimulus. Press SPACE to start measurement.'
            info_msg.draw()
            circ = visual.Circle(win=win, radius=2, pos=xy)
            circ.fillColorSpace = "rgb"
            circ.fillColor = [-1., -1., -1.]
            circ.lineColorSpace = "rgb"
            circ.lineColor = [-1., -1., -1.]
            circ.draw()
            win.flip()
            keys = event.waitKeys(keyList=['space'])

            info_msg.text = ''
            info_msg.draw()

            # start measurement
            for color in self.colors:
                for n_rep in range(6):
                    # draw stimulus
                    # get psychopy color range
                    show_color = 2. * color - 1.
                    circ.fillColorSpace = "rgb"
                    circ.fillColor = show_color
                    circ.lineColorSpace = "rgb"
                    circ.lineColor = show_color
                    circ.draw()
                    win.flip()
                    # measure spectrum
                    self.add_spectrum(name=str(color) + '#' + xy_label + '#' + str(n_rep))
        win.close()

        # set date of last measurement
        self.date = datetime.datetime.now()

    def save_to_file(self, path=None, directory=None):
        """
        Save object data to nix file.
        :param path: location of file.
        :param directory: directory, if file name should be filled automatically.
        """

        if not path:
            path = "spectrum_{}.nix".format(self.date)
        if directory:
            path = os.path.join(directory, path)
        save_dir, save_file = os.path.split(path)
        if save_dir and not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if ".nix" not in save_file:
            path = path + ".nix"

        nix_file = nix.File.open(path, mode=nix.FileMode.Overwrite)
        s = nix_file.create_section(name="meta-data", type_="meta-data")
        s.create_property(name="uuid", values_or_dtype=[str(self.uuid)])
        s.create_property(name="date", values_or_dtype=[str(self.date)])
        photometer = self.photometer.getDeviceSN() if self.photometer else ""
        s.create_property(name="photometer", values_or_dtype=[photometer])
        stepsize = self.stepsize if self.stepsize else ""
        s.create_property(name="stepsize", values_or_dtype=[stepsize])
        msp = "empty"
        if self.monitor_settings_path:
            msp = self.monitor_settings_path
            msp = str(Path(msp).resolve())
        s.create_property(name="monitor_settings_path", values_or_dtype=[msp])

        ds = nix_file.create_section(name="data", type_="data")

        for ni, name in enumerate(self.names):
            d = ds.create_section(name=str(name), type_="measurement")
            d.create_property(name="name", values_or_dtype=[str(name)])
            c = list(self.colors[ni]) if (len(self.colors) > 0) else 0.
            p_c = d.create_property(name="color", values_or_dtype=np.float64)
            p_c.values = c
            d.create_property(name="luminance", values_or_dtype=[self.spectra[name, "luminance"]])
            p_w = d.create_property(name="wavelength", values_or_dtype=np.float64)
            p_w.values = list(self.spectra[name, "wavelength"])
            p_p = d.create_property(name="power", values_or_dtype=np.float64)
            p_p.values = list(self.spectra[name, "power"])
            p_p = d.create_property(name="tristim", values_or_dtype=np.bytes_)
            p_p.values = list(self.spectra[name, "tristim"])
            p_p = d.create_property(name="uv", values_or_dtype=np.bytes_)
            p_p.values = list(self.spectra[name, "uv"])
            p_p = d.create_property(name="xy", values_or_dtype=np.bytes_)
            p_p.values = list(self.spectra[name, "xy"])
            p_p = d.create_property(name="colortemp", values_or_dtype=np.bytes_)
            p_p.values = list(self.spectra[name, "colortemp"])

        nix_file.close()

        print("Successfully saved spectra to file {}".format(path))

    def load_from_file(self, path):
        """
        Load from file.
        :param path: location of file.
        """

        nix_file = nix.File.open(path, mode=nix.FileMode.ReadOnly)
        s = nix_file.sections["meta-data"]
        self.uuid = uuid.UUID(s.props["uuid"].values[0])
        self.date = datetime.datetime.strptime(s.props["date"].values[0], "%Y%m%d")
        self.photometer = s.props["photometer"].values[0]
        self.stepsize = s.props["stepsize"].values[0]
        if s.props["monitor_settings_path"].values[0] != "empty":
            self.monitor_settings_path = s.props["monitor_settings_path"].values[0]
            self.add_monitor_settings()

        self.colors = []

        ds = nix_file.sections["data"]
        for d in ds.sections:
            name = d.props["name"].values[0]
            self.names += [name]
            self.colors += [np.asarray(d.props["color"].values)]
            self.spectra[name, "luminance"] = d.props["luminance"].values[0]
            self.spectra[name, "wavelength"] = np.asarray(d.props["wavelength"].values)
            self.spectra[name, "power"] = np.asarray(d.props["power"].values)
            self.spectra[name, "tristim"] = np.asarray(d.props["tristim"].values)
            self.spectra[name, "uv"] = np.asarray(d.props["uv"].values)
            self.spectra[name, "xy"] = np.asarray(d.props["xy"].values)
            self.spectra[name, "colortemp"] = np.asarray(d.props["colortemp"].values)

        self.colors = np.asarray(self.colors)

        nix_file.close()

        print("Successfully loaded spectra from file {}".format(path))
