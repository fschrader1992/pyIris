# -*- coding: utf-8
"""
This class contains all properties/methods related to spectra.
"""

import os
import datetime
import uuid
import json
import codecs
import numpy as np
import pandas as pd
import matplotlib.pylab as pl

from scipy.interpolate import interp1d
from symfit import variables, parameters, Model, Fit

from .spectrum import Spectrum


class Calibration:
    """
    Class Calibration
    """

    def __init__(self, corr_type="gamma_corr", mon_spectra_path="", cone_spectra_path=""):
        self.uuid = uuid.uuid4()
        self.date = datetime.datetime.now()
        self.corr_type = corr_type
        self.cone_spectra_path = cone_spectra_path
        self.mon_spectra_path = mon_spectra_path
        self._rgb_mat = None
        self._lms_mat = None
        self.calibration_matrix = np.ones((5, 3))
        self.inv_calibration_matrix = np.ones((5, 3))

        self.lum_ms = np.ones(1)
        self.lum_eff = np.ones(1)

    def rgb2lms_gamma(self, rgb):
        """
        TODO: replaced by Colorspace.rgb2lms()
        Conversion with gamma correction: lms = a_o + a * rgb**gamma.
        Uses calibration matrix.
        :param rgb: (list of) 3-tuple/numpy array with rgb values (0-1).
        :return: lms value as numpy array.
        """
        r, g, b = rgb

        min_val = 0.00000001
        r[r == 0] = min_val
        g[g == 0] = min_val
        b[b == 0] = min_val

        cm = self.calibration_matrix
        v_p = np.asarray([np.power(r, cm[4][0]), np.power(g, cm[4][1]), np.power(b, cm[4][2])])
        lms = np.tile(cm[0], (len(r), 1)).T + np.dot(cm[1:4], v_p)

        return lms

    def set_mock_values(self):
        """

        :return:
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

    def lms2rgb_gamma(self, lms):
        """
        TODO: replaced by Colorspace.lms2rgb()
        Reverse conversion from gamma correction:
        (r,g,b) = [a**I * (lms - a_0)]**(1/gamma_(r,g,b))
        Uses inverse calibration matrix.

        :param lms: (list of) 3-tuple/numpy array with lms values (0-1).
        :return: lms value as numpy array.
        """
        l, m, s = lms

        min_val = 0.00000001
        l[l == 0] = min_val
        m[m == 0] = min_val
        s[s == 0] = min_val

        cm = self.inv_calibration_matrix

        # invert and get values. in the end potentiate each list
        v_p = np.asarray([np.power(l, cm[4][0]), np.power(m, cm[4][1]), np.power(s, cm[4][2])])
        lms = np.tile(cm[0], (len(l), 1)).T + np.dot(cm[1:4], v_p)

        return lms

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
        monitor_spectra.load_from_file(path=monitor_spectra_path)

        self._rgb_mat = np.zeros(shape=(len(monitor_spectra.names), 3))
        self._lms_mat = np.zeros(shape=(len(monitor_spectra.names), 3))

        lum_eff_l = []
        lum_ms_l = []

        for i, stim in enumerate(monitor_spectra.names):

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
            self._rgb_mat[i] = np.asarray(monitor_spectra.colors[i])
            self._lms_mat[i] = np.asarray([np.trapz(l_spec * mon_spec),
                                           np.trapz(m_spec * mon_spec),
                                           np.trapz(s_spec * mon_spec)])

            # compare luminance
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
            self._lms_mat[r] = self._lms_mat[r]/max(self._lms_mat[r])

    def calibrate(self, corr_type="gamma_corr"):
        """
        Get the calibration matrix.
        :param corr_type: Type of fit/correction. Default is gamma correction.
        :return: Calibration matrix.
        """

        if corr_type != "gamma_corr":
            # here could be setting for other fit types
            raise ValueError('Correction type {} is not recognized. '
                             'Possible values are: "gamma_corr"'.format(corr_type))

        self.corr_type = corr_type

        # fit variables
        r, g, b, l, m, s = variables('r, g, b, l, m, s')

        # define models
        model = None

        if self.corr_type == "gamma_corr":
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
        min_val = 0.00000001
        r[r == 0] = min_val
        g[g == 0] = min_val
        b[b == 0] = min_val

        fit = Fit(model, r=r, g=g, b=b, l=l, m=m, s=s)
        fit_res = fit.execute()
        p_r = fit_res.params

        cm = np.ones(1)
        if self.corr_type == "gamma_corr":
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

    def plot(self):
        """
        Plot spectra rgb and lms values and fitted functions.
        Plot luminosity as well.
        Works only, if monitor spectra were obtained with color-sequence.
        """

        # RGB Values
        fig, ax = pl.subplots(ncols=2, nrows=2)

        x = np.arange(0, 1, 0.01)
        x_z = np.zeros(len(x))
        x_s = [(x, x_z, x_z), (x_z, x, x_z), (x_z, x_z, x), (x, x, x)]

        titles = ["R", "G", "B", "RGB"]

        for i in range(4):
            l_e = self._lms_mat[0][i::4]
            m_e = self._lms_mat[1][i::4]
            s_e = self._lms_mat[2][i::4]
            rgb_e = self._rgb_mat[i % 3][i::4]

            l, m, s = self.rgb2lms_gamma(x_s[i])

            ax[int(i / 2)][i % 2].plot(x, l, label="l", c="r")
            ax[int(i / 2)][i % 2].plot(x, m, label="m", c="g")
            ax[int(i / 2)][i % 2].plot(x, s, label="s", c="b")
            ax[int(i / 2)][i % 2].plot(rgb_e, l_e, "rx")
            ax[int(i / 2)][i % 2].plot(rgb_e, m_e, "gx")
            ax[int(i / 2)][i % 2].plot(rgb_e, s_e, "bx")
            ax[int(i / 2)][i % 2].set_title(titles[i])

        fig.suptitle("RGB Values")
        pl.legend()
        pl.tight_layout()
        pl.show()

        # Luminosity
        fig, ax = pl.subplots(ncols=2, nrows=2)

        x = np.arange(0, 1, 0.01)
        x_z = np.zeros(len(x))
        x_s = [(x, x_z, x_z), (x_z, x, x_z), (x_z, x_z, x), (x, x, x)]

        titles = ["R", "G", "B", "RGB"]

        for i in range(4):
            lum_ms = self.lum_ms[i::4]
            lum_eff = self.lum_eff[i::4]
            l, m, s = self.rgb2lms_gamma(x_s[i])

            rgb_e = self._rgb_mat[i % 3][i::4]

            # account for difference between measured
            # luminosity and luminance
            lum_const = 100.
            lum_calc = lum_const*(l + m)

            ax[int(i / 2)][i % 2].plot(rgb_e, lum_ms, c="k",
                                       marker="x", linestyle=":", label="Measured")
            ax[int(i / 2)][i % 2].plot(rgb_e, lum_eff, c="lightblue",
                                       marker="+", linestyle=":", label="Integrated")
            ax[int(i / 2)][i % 2].plot(x, lum_calc, "r", label="Calculated")
            ax[int(i / 2)][i % 2].set_title(titles[i])

        fig.suptitle("Luminosity")
        pl.legend()
        pl.tight_layout()
        pl.show()

    def save_to_file(self, path=None, directory=None):
        """
        Save object data to file.
        :param path: location of file.
        :param directory: directory, if file name should be filled automatically.
        """
        dt = {}
        dt.update(vars(self))
        dt["uuid"] = str(self.uuid)
        dt["date"] = str(self.date)
        dt["_rgb_mat"] = self._rgb_mat.tolist()
        dt["_lms_mat"] = self._lms_mat.tolist()
        dt["calibration_matrix"] = self.calibration_matrix.tolist()
        dt["inv_calibration_matrix"] = self.inv_calibration_matrix.tolist()
        dt["lum_eff"] = self.lum_eff.tolist()
        dt["lum_ms"] = self.lum_ms.tolist()

        if not path:
            path = "calibration_{}.json".format(self.date)
        if directory:
            path = os.path.join(directory, path)
        save_dir, save_file = os.path.split(path)
        if save_dir and not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if ".json" not in save_file:
            path = path + ".json"
        json.dump(dt, codecs.open(path, 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True, indent=4)

        print("Successfully saved calibration to file {}".format(path))

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
        self._rgb_mat = np.asarray(self._rgb_mat)
        self._lms_mat = np.asarray(self._lms_mat)
        self.calibration_matrix = np.asarray(self.calibration_matrix)
        self.inv_calibration_matrix = np.asarray(self.inv_calibration_matrix)
        self.lum_eff = np.asarray(self.lum_eff)
        self.lum_ms = np.asarray(self.lum_ms)

        print("Successfully loaded calibration from file {}".format(path))
