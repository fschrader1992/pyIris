# -*- coding: utf-8
"""
This class contains all properties/methods related to spectra.
Latest version: 2.0.0.
"""

import os
import datetime
import uuid
import json
import ruamel.yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from symfit import variables, parameters, Model, Fit

from .spectrum import Spectrum
from .functions import dump_file


class Calibration:
    """
    Class Calibration
    """

    def __init__(self, corr_type="gamma", mon_spectra_path="", cone_spectra_path="", label="main"):
        self.uuid = uuid.uuid4()
        self.date = datetime.datetime.now()
        self.corr_type = corr_type
        self.cone_spectra_path = cone_spectra_path
        self.mon_spectra_path = mon_spectra_path
        self.label = label
        self.monitor_settings_path = None
        self._rgb_mat = None
        self._lms_mat = None
        self.calibration_matrix = np.ones((5, 3))
        self.inv_calibration_matrix = np.ones((5, 3))

        self.lum_ms = np.ones(1)
        self.lum_eff = np.ones(1)
        # avoid division by zero during fitting
        self._min_val = 0.00000001

    def set_mock_values(self):
        """
        Set mock values for screensaver or test purposes.
        """
        self.calibration_matrix = np.asarray(
            [[0.0, 0.0, 0.0],
             [0.345765267310672, 0.5607281392808607, 0.12632156724602728],
             [0.07656794856797913, 0.7388016690306252, 0.22102205877927666],
             [0.007740851755583411, 0.09492307851228836, 0.9471065829653906],
             [2.080037647069668, 2.1111888095603013, 2.052076750120148]]
        )

        self.inv_calibration_matrix = np.asarray(
            [[0.0, 0.0, 0.0],
             [3.4775598238712497, -2.6595106834252373, 0.15681415600330526],
             [-0.3627818730838481, 1.672823819523023, -0.3419929663834115],
             [0.007936907301686164, -0.14592096739768426, 1.0888417086438396],
             [0.48076052921868406, 0.47366677744387564, 0.4873112079952422]]
        )

        return True

    def calc_lms_vals(self, cone_spectra_path=None, monitor_spectra_path=None):
        """
        Generate fit values for calibration.
        Integration of corresponding monitor (rgb-values) and cone spectra,
        to get respective lsm-values.
        :param cone_spectra_path: Location of path with cone spectra
        :param monitor_spectra_path: Location of path with monitor spectra.
        """

        if cone_spectra_path is None:
            cone_spectra_path = self.cone_spectra_path
        if monitor_spectra_path is None:
            monitor_spectra_path = self.mon_spectra_path

        cone_spectra = pd.read_csv(cone_spectra_path, sep=",", header=0)

        monitor_spectra = Spectrum()
        stims = []
        if ".nix" in monitor_spectra_path:
            monitor_spectra.load_from_file(path=monitor_spectra_path)
            for mn in monitor_spectra.names:
                if self.label == mn.split("#")[1]:
                    stims += [mn]
        else:
            monitor_spectra.load_from_yaml(path=monitor_spectra_path)
            stims = monitor_spectra.names
        self.monitor_settings_path = monitor_spectra.monitor_settings_path

        self._rgb_mat = np.zeros(shape=(len(monitor_spectra.names), 3))
        self._lms_mat = np.zeros(shape=(len(monitor_spectra.names), 3))

        lum_eff_l = []
        lum_ms_l = []

        for i, stim in enumerate(stims):

            # get wavelength array
            lams = np.arange(max(min(cone_spectra["wavelength"]),
                                 min(monitor_spectra.spectra[stim, "wavelength"])),
                             min(max(cone_spectra["wavelength"]),
                                 max(monitor_spectra.spectra[stim, "wavelength"])))

            # interpolate cone spectra
            l_spec = interp1d(cone_spectra["wavelength"],
                              cone_spectra["L"], kind="cubic")(lams)
            m_spec = interp1d(cone_spectra["wavelength"],
                              cone_spectra["M"], kind="cubic")(lams)
            s_spec = interp1d(cone_spectra["wavelength"],
                              cone_spectra["S"], kind="cubic")(lams)

            mon_spec = interp1d(monitor_spectra.spectra[stim, "wavelength"],
                                monitor_spectra.spectra[stim, "power"], kind="cubic")(lams)

            # stim can also be list or 3-tuple

            if ".nix" in monitor_spectra_path:
                self._rgb_mat[i] = np.asarray(monitor_spectra.spectra[stim, "RGB"])
            else:
                self._rgb_mat[i] = np.array([
                    monitor_spectra.spectra[stim, "R"],
                    monitor_spectra.spectra[stim, "G"],
                    monitor_spectra.spectra[stim, "B"],
                ])
            self._lms_mat[i] = np.asarray([np.trapz(l_spec * mon_spec),
                                           np.trapz(m_spec * mon_spec),
                                           np.trapz(s_spec * mon_spec)])

            # compare luminosity
            lum_eff = np.trapz(l_spec * mon_spec) + np.trapz(m_spec * mon_spec)
            delta = lum_eff - monitor_spectra.spectra[stim, "luminance"]
            efc = delta / monitor_spectra.spectra[stim, "luminance"]

            lum_eff_l += [lum_eff]
            lum_ms_l += [monitor_spectra.spectra[stim, "luminance"]]

        self.lum_eff = np.asarray(lum_eff_l)
        self.lum_ms = np.asarray(lum_ms_l)

        self._lms_mat = self._lms_mat.T
        self._rgb_mat = self._rgb_mat.T

        for r in (0, 1, 2):
            self._lms_mat[r] = self._lms_mat[r]/max(self._lms_mat[r]) * np.max(self._rgb_mat)

        return True

    def calibrate(self, corr_type="gamma"):
        """
        Get the calibration matrix.
        :param corr_type: Type of fit/correction. Default is gamma correction.
        :return: Calibration matrix.
        """

        if corr_type != "gamma":
            # here could be setting for other fit types
            raise ValueError('Correction type {} is not recognized. '
                             'Possible values are: "gamma"'.format(corr_type))

        self.corr_type = corr_type

        # fit variables
        r, g, b, l, m, s = variables('r, g, b, l, m, s')

        # define models
        model = None

        if self.corr_type == "gamma":
            a_0l, a_0m, a_0s = parameters('a_0l, a_0m, a_0s', min=0.0, value=0.0)
            a_lr, a_lg, a_lb, a_mr, a_mg, a_mb, a_sr, a_sg, a_sb = \
                parameters('a_lr, a_lg, a_lb, a_mr, a_mg, a_mb, '
                           'a_sr, a_sg, a_sb', min=0.0, value=1.0)
            gamma_r, gamma_g, gamma_b = parameters('gamma_r, gamma_g, gamma_b', value=1.5)

            model = Model({
                l: a_0l + a_lr * r ** gamma_r + a_lg * g ** gamma_g + a_lb * b ** gamma_b,
                m: a_0m + a_mr * r ** gamma_r + a_mg * g ** gamma_g + a_mb * b ** gamma_b,
                s: a_0s + a_sr * r ** gamma_r + a_sg * g ** gamma_g + a_sb * b ** gamma_b,
            })

        # get values for variables
        r, g, b = self._rgb_mat
        l, m, s = self._lms_mat

        # avoid division by zero errors
        r[r == 0] = self._min_val
        g[g == 0] = self._min_val
        b[b == 0] = self._min_val

        fit = Fit(model, r=r, g=g, b=b, l=l, m=m, s=s)
        fit_res = fit.execute()
        p_r = fit_res.params

        cm = np.ones(1)
        if self.corr_type == "gamma":
            cm = np.asarray([[p_r["a_0l"], p_r["a_0m"], p_r["a_0s"]],
                             [p_r["a_lr"], p_r["a_lg"], p_r["a_lb"]],
                             [p_r["a_mr"], p_r["a_mg"], p_r["a_mb"]],
                             [p_r["a_sr"], p_r["a_sg"], p_r["a_sb"]],
                             [p_r["gamma_r"], p_r["gamma_g"], p_r["gamma_b"]]])

        self.calibration_matrix = cm

        inv_mat = np.zeros((5, 3))
        inv_mat[0] = cm[0]
        inv_mat[1:4] = np.linalg.inv(cm[1:4])
        inv_mat[4] = np.asarray([1./cm[4][0], 1./cm[4][1], 1./cm[4][2]])

        self.inv_calibration_matrix = inv_mat

        return True

    def pprint(self):
        """
        Print data to CLI.
        """
        print(vars(self))

        print("CALIBRATION VALUES (LMS)")
        print("r\tg\tb")

        if self.corr_type == "gamma corr":
            cm = self.calibration_matrix
            a_0 = cm[0]
            a = cm[1:4]
            gamma = cm[4]

            print("a_0\n", a_0)
            print("a\n", a)
            print("gamma", gamma)

        return True

    def plot(self, path=None, directory=None, show=True):
        """
        Plot spectra rgb and lms values and fitted functions.
        Plot luminosity as well.
        Works only, if monitor spectra were obtained with color-sequence.
        :param path: Path to file. Default is None, which results in plot_calibration_<DATE>.pdf file.
        :param directory: Directory, prepended to "path".
                         Default is None, which creates a directory called "calibration_plots" in the current directory.
        :param show: If True, plot will be shown, otherwise only saved. Default is True.
        """

        try:
            from .colorspace import ColorSpace
        except ImportError:
            pass

        # save file options
        if not path:
            path = "calibration_plots/plot_calibration_{}".format(self.date)
        if directory:
            path = os.path.join(directory, os.path.basename(path))
        os.makedirs(os.path.split(path)[0], exist_ok=True)

        # Fit Values
        fig, ax = plt.subplots(ncols=3, nrows=7, figsize=(10, 16))
        cs = ColorSpace()
        cs.calibration = self
        x = np.arange(0., 1., 0.01)
        x_z = np.zeros(len(x))
        x_s = [(x, x_z, x_z), (x_z, x, x_z), (x_z, x_z, x), (x, x, x_z), (x, x_z, x), (x_z, x, x), (x, x, x)]

        titles = ["R", "G", "B", "RG", "RB", "GB", "RGB"]

        ch_rgb_mat = np.where(self._rgb_mat > self._min_val, 1, 0).T
        ch_rgb_mat = np.where(ch_rgb_mat < 0.001, 0, 1)
        ch_x_s = np.where(np.asarray(x_s) > 0, 1, 0).T[-1].T

        for i in range(len(x_s)):
            inds = np.argwhere(np.all(np.equal(ch_rgb_mat, ch_x_s[i]), axis=1))
            l_e = self._lms_mat[0][inds].T[0]
            m_e = self._lms_mat[1][inds].T[0]
            s_e = self._lms_mat[2][inds].T[0]
            lms_e = np.asarray([l_e, m_e, s_e]).T
            rgb_e = np.max(self._rgb_mat.T[inds].T, axis=0)[0]

            lms = cs.rgb2lms(np.asarray(x_s[i]).T)
            l, m, s = lms.T
            ax[i][0].grid()
            ax[i][1].grid()
            ax[i][2].grid()
            ax[i][0].plot(x, l, label="L", c="tab:red", linewidth=1)
            ax[i][0].plot(x, m, label="M", c="tab:green", linewidth=1)
            ax[i][0].plot(x, s, label="S", c="tab:blue", linewidth=1)
            ax[i][0].plot(rgb_e, l_e, c="tab:red", marker="x", linestyle=None)
            ax[i][0].plot(rgb_e, m_e, c="tab:green", marker="x", linestyle=None)
            ax[i][0].plot(rgb_e, s_e, c="tab:blue", marker="x", linestyle=None)
            ax[i][0].set_title(titles[i] + ", LMS-Values")

            ha, sat = cs.lms2cnop(lms, x)
            ha_e, sat_e = cs.lms2cnop(lms_e, rgb_e)

            ax[i][1].plot(x, ha % 360., label="Fit", c="tab:blue", linewidth=1)
            ax[i][2].plot(x, sat, label="Fit", c="tab:blue", linewidth=1)
            ax[i][1].plot(rgb_e, ha_e % 360., label="Data", c="tab:blue", marker="x", linestyle=None)
            ax[i][2].plot(rgb_e, sat_e, label="Data", c="tab:blue", marker="x", linestyle=None)
            ax[i][1].set_title(titles[i] + ", Hue Angle")
            ax[i][2].set_title(titles[i] + ", Saturation")
            if i == 0:
                ax[i][0].legend()
                ax[i][1].legend()
                ax[i][2].legend()
        fig.suptitle("Calibration Data and Fits")
        fig.text(0.5, 0.0, "Ratio Component Intensity", va="bottom", ha="center", size=12)
        plt.tight_layout()
        plt.savefig(path + "_Fit.pdf")
        if show:
            plt.show()
        plt.cla()

        # Luminance
        fig, ax = plt.subplots(ncols=3, nrows=3, sharex=True, figsize=(10, 8))

        x = np.arange(0., 1., 0.01)
        x_z = np.zeros(len(x))
        x_s = [(x, x_z, x_z), (x_z, x, x_z), (x_z, x_z, x), (x, x, x_z), (x, x_z, x), (x_z, x, x), (x, x, x)]

        for i in range(len(x_s)):
            inds = np.argwhere(np.all(np.equal(ch_rgb_mat, ch_x_s[i]), axis=1))
            lum_ms = self.lum_ms[inds]
            lum_eff = self.lum_eff[inds]
            l, m, s = cs.rgb2lms(np.asarray(x_s[i]).T).T

            rgb_e = np.max(self._rgb_mat.T[inds].T, axis=0)[0]

            # account for difference between measured
            # luminosity and luminance
            lum_const = 100.
            lum_calc = lum_const*(l + m)

            ax[int(i / 3)][i % 3].plot(rgb_e, lum_ms, c="k", linewidth=1,
                                       marker="x", label="Measured")
            ax[int(i / 3)][i % 3].plot(rgb_e, lum_eff, c="tab:blue", linewidth=1,
                                       marker="+", label="Integrated")
            ax[int(i / 3)][i % 3].plot(x, lum_calc, linewidth=1, c="tab:red", label="Calculated")
            ax[int(i / 3)][i % 3].set_title(titles[i])
            if i == 0:
                ax[int(i / 3)][i % 3].legend()

        fig.suptitle("Luminance")
        plt.tight_layout()
        plt.savefig(path + "_luminance.pdf")
        if show:
            plt.show()
        plt.cla()

        return True

    def plot_differences(self, path=None, directory=None, show=True):
        """
        Plot spectra rgb and lms values and fitted functions.
        Plot luminosity as well.
        Works only, if monitor spectra were obtained with color-sequence.
        :param path: Path to file. Default is None, which results in plot_calibration_<DATE>.pdf file.
        :param directory: Directory, prepended to "path".
                         Default is None, which creates a directory called "calibration_plots" in the current directory.
        :param show: If True, plot will be shown, otherwise only saved. Default is True.
        """

        try:
            from .colorspace import ColorSpace
        except ImportError:
            pass

        # save file options
        if not path:
            path = "calibration_plots/plot_calibration_{}".format(self.date)
        if directory:
            path = os.path.join(directory, os.path.basename(path))
        os.makedirs(os.path.split(path)[0], exist_ok=True)

        # RGB Values
        fig, ax = plt.subplots(ncols=3, nrows=7, figsize=(10, 16))
        cs = ColorSpace()
        cs.calibration = self
        x = np.arange(0, 1, 0.01)
        x_z = np.zeros(len(x))
        x_s = [(x, x_z, x_z), (x_z, x, x_z), (x_z, x_z, x), (x, x, x_z), (x, x_z, x), (x_z, x, x), (x, x, x)]

        titles = ["R", "G", "B", "RG", "RB", "GB", "RGB"]

        ch_rgb_mat = np.where(self._rgb_mat > self._min_val, 1, 0).T
        ch_rgb_mat = np.where(ch_rgb_mat < 0.001, 0, 1)
        ch_x_s = np.where(np.asarray(x_s) > 0, 1, 0).T[-1].T

        for i in range(len(x_s)):
            inds = np.argwhere(np.all(np.equal(ch_rgb_mat, ch_x_s[i]), axis=1))
            l_e = self._lms_mat[0][inds].T[0]
            m_e = self._lms_mat[1][inds].T[0]
            s_e = self._lms_mat[2][inds].T[0]
            lms_e = np.asarray([l_e, m_e, s_e]).T
            rgb_e = np.max(self._rgb_mat.T[inds].T, axis=0)[0]

            x = rgb_e
            x_z = np.zeros(len(x))
            x_s = [(x, x_z, x_z), (x_z, x, x_z), (x_z, x_z, x), (x, x, x_z), (x, x_z, x), (x_z, x, x), (x, x, x)]
            lms = cs.rgb2lms(np.asarray(x_s[i]).T)
            l, m, s = lms.T
            ax[i][0].grid()
            ax[i][1].grid()
            ax[i][2].grid()
            ax[i][0].plot(rgb_e, l - l_e, label="L, Fit - Data", c="tab:red", marker="x", linewidth=1)
            ax[i][0].plot(rgb_e, m - m_e, label="M, Fit - Data", c="tab:green", marker="x", linewidth=1)
            ax[i][0].plot(rgb_e, s - s_e, label="S, Fit - Data", c="tab:blue", marker="x", linewidth=1)
            ax[i][0].set_title(titles[i] + ", LMS-Values")

            ha, sat = cs.lms2cnop(lms, rgb_e)
            ha_e, sat_e = cs.lms2cnop(lms_e, rgb_e)

            ax[i][1].plot(rgb_e, ha - ha_e, label="HA, Fit - Data", c="tab:blue", marker="x", linewidth=1)
            ax[i][2].plot(rgb_e, sat - sat_e, label="SAT, Fit - Data", c="tab:blue", marker="x", linewidth=1)
            ax[i][1].set_title(titles[i] + ", mean HA: " + str(np.round(np.mean(ha_e), 2) % 360.))
            ax[i][2].set_title(titles[i] + ", mean SAT: " + str(np.round(np.mean(sat_e), 3)))
            if i == 0:
                ax[i][0].legend()
                ax[i][1].legend()
                ax[i][2].legend()
        fig.suptitle("Calibration Fit - Data")
        fig.text(0.5, 0.0, "Ratio Component Intensity", va="bottom", ha="center", size=12)
        plt.tight_layout()
        plt.savefig(path + "_differences.pdf")
        if show:
            plt.show()
        plt.cla()


        # Luminance Differences
        fig, ax = plt.subplots(ncols=3, nrows=3, sharex=True, figsize=(10, 8))

        x = np.arange(0, 1, 0.01)
        x_z = np.zeros(len(x))
        x_s = [(x, x_z, x_z), (x_z, x, x_z), (x_z, x_z, x), (x, x, x_z), (x, x_z, x), (x_z, x, x), (x, x, x)]
        for i in range(len(x_s)):
            inds = np.argwhere(np.all(np.equal(ch_rgb_mat, ch_x_s[i]), axis=1))
            lum_ms = self.lum_ms[inds].T[0]
            lum_eff = self.lum_eff[inds].T[0]

            rgb_e = np.max(self._rgb_mat.T[inds].T, axis=0)[0]
            x = rgb_e
            x_z = np.zeros(len(x))
            x_si = [(x, x_z, x_z), (x_z, x, x_z), (x_z, x_z, x), (x, x, x_z), (x, x_z, x), (x_z, x, x), (x, x, x)]
            l, m, s = cs.rgb2lms(np.asarray(x_si[i]).T).T

            # account for difference between measured
            # luminosity and luminance
            lum_const = 100.
            lum_calc = lum_const*(l + m)

            ax[int(i / 3)][i % 3].plot(x, lum_eff-lum_ms, c="tab:blue", linewidth=1,
                                       marker="x", label="Eff. - Meas.")
            ax[int(i / 3)][i % 3].plot(x, lum_calc-lum_ms, c="tab:orange", linewidth=1,
                                       marker="x", label="Calc. - Meas.")
            ax[int(i / 3)][i % 3].plot(x, lum_calc-lum_eff, linewidth=1, c="tab:green", marker="x", label="Calc. - Eff.")
            ax[int(i / 3)][i % 3].set_title(titles[i])
            if i == 0:
                ax[int(i / 3)][i % 3].legend()

        fig.suptitle("Luminance Differences")
        plt.tight_layout()
        plt.savefig(path + "_luminance_differences.pdf")
        if show:
            plt.show()
        plt.cla()

        return True

    def save_to_file(self, path=None, directory=None, filetype="yaml", absolute_paths=False):
        """
        Save calibration data to file.
        :param path: Location of file. Default in None.
        :param directory: Directory, if file name should be filled automatically.
        :param filetype: Filetype, "json" or "yaml".
               Default is "yaml" but set to file extension if found in path.
        :param absolute_paths: If True, absolute file paths are saved. Default is False.
        :return: True.
        """

        if path is not None and "." in path:
            filetype = path.split(".")[-1]

        dt = {}
        dt.update(vars(self))
        dt["uuid"] = str(self.uuid)
        dt["date"] = self.date.isoformat()
        dt["_rgb_mat"] = self._rgb_mat.tolist()
        dt["_lms_mat"] = self._lms_mat.tolist()
        dt["calibration_matrix"] = self.calibration_matrix.tolist()
        dt["inv_calibration_matrix"] = self.inv_calibration_matrix.tolist()
        dt["lum_eff"] = self.lum_eff.tolist()
        dt["lum_ms"] = self.lum_ms.tolist()
        if "cone_spectra_path" in dt.keys():
            if absolute_paths:
                dt["cone_spectra_path"] = os.path.abspath(dt["cone_spectra_path"])
        if "mon_spectra_path" in dt.keys():
            if absolute_paths:
                dt["mon_spectra_path"] = os.path.abspath(dt["mon_spectra_path"])
        if dt["monitor_settings_path"]:
            if absolute_paths:
                dt["monitor_settings_path"] = os.path.abspath(dt["monitor_settings_path"])
        dt = dict(sorted(dt.items()))

        if not path:
            path = "calibration_{}.{}".format(self.date, filetype)
        if directory:
            path = os.path.join(directory, path)
        dump_file(dt, path, filetype)

        print("Successfully saved calibration to file {}".format(path))

        return True

    def load_from_file(self, path=None, filetype="yaml"):
        """
        Load calibration from file.
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
        for a, b in d.items():
            setattr(self, a, self.__class__(b) if isinstance(b, dict) else b)

        self.uuid = uuid.UUID(str(self.uuid))
        self.date = datetime.datetime.fromisoformat(d["date"])
        self._rgb_mat = np.asarray(self._rgb_mat)
        self._lms_mat = np.asarray(self._lms_mat)
        self.calibration_matrix = np.asarray(self.calibration_matrix)
        self.inv_calibration_matrix = np.asarray(self.inv_calibration_matrix)
        self.lum_eff = np.asarray(self.lum_eff)
        self.lum_ms = np.asarray(self.lum_ms)

        print("Successfully loaded calibration from file {}".format(path))

        return True
