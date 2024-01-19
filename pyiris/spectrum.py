# -*- coding: utf-8
"""
This class contains all properties/methods related to spectra.
Latest version: 2.0.0.
"""

import os
import datetime
import uuid
import h5py
import ruamel.yaml
import matplotlib.pyplot as plt
import numpy as np
import nixio as nix

from pathlib import Path
from psychopy import logging, visual, event

from .pr655 import PR655
from .monitor import Monitor
from .functions import prepare_for_yaml


class Spectrum:
    """
    Class Spectrum
    """

    def __init__(self, photometer=None, colors=None, monitor_settings_path=None, path=None):
        self.uuid = uuid.uuid4()
        self.date = datetime.datetime.now()
        self.photometer = photometer
        self.colors = colors
        self.path = path if path is not None else "spectrum_{}".format(self.date.isoformat(timespec="seconds"))

        self.monitor_settings_path = monitor_settings_path
        self.monitor = None

        self.params = {}
        self.names = []
        self.spectra = {}

        # print errors
        logging.console.setLevel(logging.ERROR)

        if colors:
            self.names = colors
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

        return True

    def create_colorlist(self, stepsize=0.1, minval=0., maxval=1.):
        """
        Set list of rgb-color codes to be used for spectra.
        :param stepsize: Difference between two color steps (between 0. and 1.)
        """
        self.colors = []
        for step in np.arange(minval, maxval + stepsize, stepsize):
            self.colors += [np.asarray([step, 0., 0.])]
            self.colors += [np.asarray([0., step, 0.])]
            self.colors += [np.asarray([0., 0., step])]
            self.colors += [np.asarray([step, step, 0.])]
            self.colors += [np.asarray([0., step, step])]
            self.colors += [np.asarray([step, 0., step])]
            self.colors += [np.asarray([step, step, step])]

        return True

    def add_pr655(self, port="/dev/ttyUSB0"):
        """
        Automatically add PR655, if connected.
        """
        self.photometer = PR655(port=port)

        return True

    def add_spectrum(self, name, xy, rgb, label, repeat, save_append=True):
        """
        Measure and save spectrum from photometer.
        :param name: Name this measurement has. With measure_colors this is equal to rgb-color code.
        """
        self.names += [name]
        self.spectra[name, "label"] = label
        self.spectra[name, "repeat"] = repeat
        self.spectra[name, "screen-pos-x"] = xy[0]
        self.spectra[name, "screen-pos-y"] = xy[1]
        self.spectra[name, "R"] = rgb[0]
        self.spectra[name, "G"] = rgb[1]
        self.spectra[name, "B"] = rgb[2]
        self.spectra[name, "luminance"] = self.photometer.getLum()
        nm, power = self.photometer.getLastSpectrum(parse=True)
        self.spectra[name, "wavelength"] = nm
        self.spectra[name, "power"] = power

        # get other data
        self.spectra[name, "tristim"] = self.photometer.getLastTristim()
        self.spectra[name, "uv"] = self.photometer.getLastUV()
        self.spectra[name, "xy"] = self.photometer.getLastXY()
        self.spectra[name, "colortemp"] = self.photometer.getLastColorTemp()
        self.spectra[name, "date"] = datetime.datetime.now().isoformat()

        if save_append:
            self.save_addend_to_yaml(name)

        return True

    def add_mock_spectrum(self, name, i):
        """
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

        return True

    def measure_colors(self, stepsize=0.1, minval=0., maxval=1., n_rep=1,
                       xys=None, xy_labels=None,
                       stim_type=None, stim_size=None, background=0.67,
                       win_h=1200, win_w=1800, save_append=True):
        """
        Measure the spectra for each color stimulus, for stimuli
        in different areas of the screen.
        """

        # Fill self.colorlist
        self.create_colorlist(stepsize=stepsize, minval=minval, maxval=maxval)
        # obtain number of measurements
        noc = len(self.colors) * n_rep
        colors_updated = []

        # define background color during measurements
        if isinstance(background, float) or isinstance(background, int):
            background = np.array([background, background, background])
        elif (isinstance(background, list) and len(background) == 1) or\
                (isinstance(background, np.ndarray) and len(background) == 1):
            background = np.array([background[0], background[0], background[0]])
        background = 2. * background - 1.
        # define background during photometer adjustments
        info_background = [0.33, 0.33, 0.33]

        # define window
        win = visual.Window([win_h, win_w], color=info_background, fullscr=True)
        if self.monitor:
            win.monitor = self.monitor

        # create info message stimulus
        info_msg = visual.TextStim(win, '', color=[1., 1., 1.], pos=(0, 10), height=0.75, units='deg')

        # Positions: use center of monitor by default
        if xys is None:
            xys = [[0., 0.]]
            xy_labels = ['main']

        # Define stimulus for measurements
        measure_stim = None
        if stim_type == "circ" or stim_type == "circle":
            measure_stim = visual.Circle(win=win, radius=1, pos=[0., 0.], units='deg')
        else:
            measure_stim = visual.Rect(win=win, width=win_w, height=win_h)
        if stim_size is not None:
            measure_stim.size = stim_size

        # store function input for saving
        self.params['stepsize'] = stepsize
        self.params['stim_type'] = stim_type if stim_type is not None else 'rectangle'
        self.params['stim_size'] = stim_size if stim_size is not None else 0.
        self.params['background'] = background
        self.params['win-units'] = "deg"

        if save_append:
            self.save_as_yaml()

        # iterate through positions
        for xy_label, xy in zip(xy_labels, xys):
            # Define window background color during adjustment
            win.color = info_background
            # start with stimulus in order to adjust photometer
            info_msg.color = [1., 1., 1.]
            info_msg.text = 'Please adjust the photometer to the stimulus. Press SPACE to start measurement.'
            info_msg.draw()
            circ = visual.Circle(win=win, radius=1, pos=xy, units='deg')
            circ.fillColorSpace = "rgb"
            circ.fillColor = [1., 1., 1.]
            circ.lineColorSpace = "rgb"
            circ.lineColor = [1., 1., 1.]
            circ.draw()
            win.flip()
            keys = event.waitKeys(keyList=['space'])

            # Define window background color
            win.color = background
            info_msg.color = [-0.5, -0.5, -0.5]

            # start measurement
            q = 1
            for color in self.colors:
                # get psychopy color range
                show_color = 2. * color - 1.
                for n in range(n_rep):
                    # add same color to color list to save correctly
                    colors_updated += [color]
                    # draw stimulus
                    measure_stim.fillColor = show_color
                    measure_stim.lineColor = show_color
                    measure_stim.draw()
                    info_msg.text = str(q) + '/' + str(noc)
                    info_msg.draw()
                    win.flip()
                    # measure spectrum
                    self.add_spectrum(name="{}#{}#{}".format(str(color), xy_label, n+1),
                                      xy=xy, rgb=color, label=xy_label, repeat=n+1, save_append=save_append)
                    q += 1
        win.close()

        # update color list
        self.colors = colors_updated

        return True

    def plot_spectra(self, path=None, show=True):
        """
        Plot measured spectra.
        :param path: Path to file.
        :param show: If True, plot will be shown, otherwise only saved. Default is True.
        """

        # save file options
        if not path:
            path = "measured_spectra_{}.pdf".format(self.date)
        plot_dir = "calibration_plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        path = os.path.join(plot_dir, path)

        plot_dict = {
            (1., 0., 0.): {"label": "R", "plot_color": "tab:red", "ax_ind": 0},
            (0., 1., 0.): {"label": "G", "plot_color": "tab:green", "ax_ind": 1},
            (0., 0., 1.): {"label": "B", "plot_color": "tab:blue", "ax_ind": 2},
            (1., 1., 0.): {"label": "RG", "plot_color": "tab:orange", "ax_ind": 3},
            (1., 0., 1.): {"label": "RB", "plot_color": "tab:purple", "ax_ind": 4},
            (0., 1., 1.): {"label": "GB", "plot_color": "tab:cyan", "ax_ind": 5},
            (1., 1., 1.): {"label": "RGB", "plot_color": "k", "ax_ind": 6},
        }

        fig, ax = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True, figsize=(10, 8))

        added_titles = []
        for nm in self.names:
            pattern = (self.spectra[nm, "R"], self.spectra[nm, "G"], self.spectra[nm, "B"])
            # u = float(self.spectra[nm, "uv"][3])
            # v = float(self.spectra[nm, "uv"][4].replace("\n", "").replace("\r", ""))
            x = self.spectra[nm, "wavelength"]
            pow = self.spectra[nm, "power"]
            a = np.max(pattern)
            pattern = np.sign(np.asarray(pattern))
            pattern = (pattern[0], pattern[1], pattern[2])
            axi = plot_dict[pattern]["ax_ind"]
            ax[int(axi / 3)][axi % 3].plot(x, pow, c=plot_dict[pattern]["plot_color"], linewidth=1, alpha=a)
            # ax[2][1].plot(u, v, c=plot_dict[pattern]["plot_color"], marker="o", alpha=a)
            if plot_dict[pattern]["label"] not in added_titles:
                ax[int(axi / 3)][axi % 3].set_title(plot_dict[pattern]["label"])
                added_titles += [plot_dict[pattern]["label"]]

        fig.suptitle("Measured Spectra")
        fig.text(0.5, 0.0, "Wavelength [nm]", va="bottom", ha="center", size=12)
        fig.text(0.01, 0.5, "Radiance", rotation=90, va="bottom", ha="center", size=12)
        fig.tight_layout()
        plt.savefig(path)
        if show:
            plt.show()
        plt.cla()

        return True

    def save_to_file(self, path=None, directory=None):
        """
        Save object data to nix file.
        :param path: location of file.
        :param directory: directory, if file name should be filled automatically.
        """

        if not path:
            path = self.path
        path = path.replace(".yaml", "").replace(".yml", "")
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
        try:
            photometer = int(self.photometer.getDeviceSN())
        except AttributeError:
            photometer = 0
        s.create_property(name="photometer", values_or_dtype=[photometer])
        msp = "empty"
        if self.monitor_settings_path:
            msp = self.monitor_settings_path
            msp = str(Path(msp).resolve())
        s.create_property(name="monitor_settings_path", values_or_dtype=[msp])

        # save params
        s.create_property(name="stepsize", values_or_dtype=[self.params["stepsize"]])
        s.create_property(name="stimsize", values_or_dtype=[self.params["stim_size"]])
        s.create_property(name="stimtype", values_or_dtype=[self.params["stim_type"]])
        bg = list(self.params["background"]) if (len(self.params["background"]) > 0) else 0.
        p_bg = s.create_property(name="background", values_or_dtype=np.float64)
        p_bg.values = bg

        ds = nix_file.create_section(name="data", type_="data")

        for ni, name in enumerate(self.names):
            d = ds.create_section(name=str(name), type_="measurement")
            d.create_property(name="name", values_or_dtype=[str(name)])
            d.create_property(name="label", values_or_dtype=[self.spectra[name, "label"]])
            d.create_property(name="repeat", values_or_dtype=[self.spectra[name, "repeat"]])
            d.create_property(name="screen-pos-x", values_or_dtype=[self.spectra[name, "screen-pos-x"]])
            d.create_property(name="screen-pos-y", values_or_dtype=[self.spectra[name, "screen-pos-y"]])
            d.create_property(name="R", values_or_dtype=[self.spectra[name, "R"]])
            d.create_property(name="G", values_or_dtype=[self.spectra[name, "G"]])
            d.create_property(name="B", values_or_dtype=[self.spectra[name, "B"]])
            d.create_property(name="luminance", values_or_dtype=[self.spectra[name, "luminance"]])
            p_w = d.create_property(name="wavelength", values_or_dtype=np.float64)
            p_w.values = list(self.spectra[name, "wavelength"])
            p_p = d.create_property(name="power", values_or_dtype=np.float64)
            p_p.values = list(self.spectra[name, "power"])
            p_p = d.create_property(name="tristim", values_or_dtype=np.str_)
            p_p.values = list(self.spectra[name, "tristim"])
            p_p = d.create_property(name="uv", values_or_dtype=np.str_)
            p_p.values = list(self.spectra[name, "uv"])
            p_p = d.create_property(name="xy", values_or_dtype=np.str_)
            p_p.values = list(self.spectra[name, "xy"])
            p_p = d.create_property(name="colortemp", values_or_dtype=np.str_)
            p_p.values = list(self.spectra[name, "colortemp"])

        nix_file.close()

        print("Successfully saved spectra to file {}".format(path))

        return True

    def save_as_yaml(self, path=None, directory=None):
        """
        Save spectrum data to yaml file.
        :param path: Location of file.
        :param directory: Directory, if file name should be filled automatically.
        """

        if not path:
            path = self.path
        path = path.replace(".nix", "")
        if directory:
            path = os.path.join(directory, path)
        save_dir, save_file = os.path.split(path)
        if save_dir and not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if ".yaml" not in save_file:
            path = path + ".yaml"

        dt = dict({})
        md = dict({})

        # add metadata
        md.update(vars(self))
        md["date"] = self.date.isoformat()
        md["uuid"] = str(self.uuid)
        try:
            md["photometer"] = int(self.photometer.getDeviceSN())
        except AttributeError:
            md["photometer"] = 0
        del md["monitor"]
        del md["path"]
        del md["spectra"]

        # sort keys
        dt["metadata"] = dict(sorted(md.items()))

        # add spectral data
        # create dict with 2 layers from tuple keys to make file more readable
        sd = dict.fromkeys(self.names, None)
        for sk, sv in self.spectra.items():
            if sd[sk[0]] is None:
                sd[sk[0]] = dict({})
            if sk[1] in ["tristim", "uv", "xy", "colortemp"]:
               sv = list(map(lambda v: str(v).replace("\n", "").replace("\r", ""), sv))
            sd[sk[0]][sk[1]] = sv

        dt.update(sd)
        # process trial data
        dt = prepare_for_yaml(dt, list_compression=True)

        # write datafile
        with open(path, "w") as outfile:
            ruamel.yaml.YAML().dump(dt, outfile)

        print("Successfully saved spectra to yaml-file {}".format(path))
        return True

    def save_addend_to_yaml(self, name, path=None, directory=None):
        """
        Append spectrum data to yaml file.
        :param name: Spectrum measurement name.
        :param path: Location of file.
        :param directory: Directory, if file name should be filled automatically.
        """

        if not path:
            path = self.path.replace(".nix", "")
        if directory:
            path = os.path.join(directory, path)
        save_dir, save_file = os.path.split(path)
        if ".yaml" not in save_file:
            path = path + ".yaml"

        sd = dict({name: dict({})})
        for sk, sv in self.spectra.items():
            if sk[0] != name:
                continue
            sv = self.spectra[sk]
            if sk[1] in ["tristim", "uv", "xy", "colortemp"]:
                sv = list(map(lambda v: str(v).replace("\n", "").replace("\r", ""), sv))
            sd[sk[0]][sk[1]] = sv

        # process trial data
        sd = prepare_for_yaml(sd, list_compression=True)

        # write datafile
        with open(path, "a+") as outfile:
            ruamel.yaml.YAML().dump(sd, outfile)

        return True

    def load_from_file(self, path):
        """
        Load from file.
        :param path: location of file.
        """

        if ".nix" not in path:
            path += ".nix"
        self.path = path
        nix_file = nix.File.open(path, mode=nix.FileMode.ReadOnly)
        s = nix_file.sections["meta-data"]
        self.uuid = uuid.UUID(s.props["uuid"].values[0])
        # catch older versions with different date format
        try:
            self.date = datetime.datetime.strptime(s.props["date"].values[0], "%Y%m%d")
        except ValueError:
            try:
                self.date = datetime.datetime.fromisoformat(s.props["date"].values[0])
            except ValueError:
                self.date = None
                pass
            pass

        self.photometer = s.props["photometer"].values[0]
        if self.photometer == 0:
            self.photometer = None
        if s.props["monitor_settings_path"].values[0] != "empty":
            self.monitor_settings_path = s.props["monitor_settings_path"].values[0]
            try:
                self.add_monitor_settings()
            except FileNotFoundError:
                print('Error, monitor settings file', self.monitor_settings_path, 'could not be found and is skipped.')

        self.colors = []

        ds = nix_file.sections["data"]
        for d in ds.sections:
            name = d.props["name"].values[0]
            self.names += [name]
            self.spectra[name, "label"] = d.props["label"].values[0]
            self.spectra[name, "repeat"] = d.props["repeat"].values[0]
            self.spectra[name, "R"] = d.props["R"].values[0]
            self.spectra[name, "G"] = d.props["G"].values[0]
            self.spectra[name, "B"] = d.props["B"].values[0]
            self.spectra[name, "RGB"] = [np.array([
                self.spectra[name, "R"], self.spectra[name, "G"], self.spectra[name, "B"]
            ])]
            self.spectra[name, "screen-pos-x"] = d.props["screen-pos-x"].values[0]
            self.spectra[name, "screen-pos-y"] = d.props["screen-pos-y"].values[0]
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

        return True

    def load_from_yaml(self, path):
        """
        Load from yaml file.
        :param path: Location of file.
        """

        if ".yaml" not in path and ".yml" not in path:
            path += ".yaml"
        self.path = path
        with open(path, "r") as f:
            sd = ruamel.yaml.YAML().load(f)
        for a, b in sd["metadata"].items():
            setattr(self, a, b)
        self.uuid = uuid.UUID(self.uuid)
        self.date = datetime.datetime.fromisoformat(self.date)
        if len(self.monitor_settings_path) > 0:
            try:
                self.add_monitor_settings()
            except FileNotFoundError:
                print('Error, monitor settings file', self.monitor_settings_path, 'could not be found and is skipped.')

        del sd['metadata']
        names_from_keys = True if len(self.names) == 0 else False

        for sk0 in sd.keys():
            if names_from_keys:
                self.names += [sk0]
            for sk1 in sd[sk0].keys():
                self.spectra[sk0, sk1] = sd[sk0][sk1]

        print("Successfully loaded spectra from file {}".format(path))

        return True

    def yaml2nix(self, path, save_path=None):
        """
        Convert a spectrum file saved in yaml format to a nix file.
        :param path: Location of file.
        :param save_path: Path to save converted spectrum file to.
                          Default is None, which stores spectrum under the same name with changed extension.
        """

        if save_path is None:
            save_path = path
        self.load_from_yaml(path)
        self.save_to_file(save_path)

        return True

    def nix2yaml(self, path, save_path=None):
        """
        Convert a spectrum file saved in nix format to a yaml file.
        :param path: Location of file.
        :param save_path: Path to save converted spectrum file to.
                          Default is None, which stores spectrum under the same name with changed extension.
        """

        if save_path is None:
            save_path = path
        self.load_from_file(path)
        self.save_as_yaml(save_path)

        return True
