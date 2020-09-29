# -*- coding: utf-8
"""
This class contains all properties/methods related to color spaces.
"""

import os
import datetime
import uuid
import json
import codecs
import numpy as np
import matplotlib.pylab as pl

from psychopy import event, misc, visual
from scipy import optimize

from .subject import Subject
from .calibration import Calibration


def sine_fitter(x, amp, phi):
    """
    For iso-slant fit.
    :param x: Hue angle.
    :param amp: Amplitude.
    :param phi: Phase.
    :return: Sine value.
    """
    return amp * np.sin(x + phi)


class ColorSpace:
    """
    Class Colorspace
    """

    def __init__(self, calibration_path=None, subject_path=None, bit_depth=10, chromaticity=0.12,
                 gray_lavel=0.66, unit="rad", s_scale=2.6):

        self.uuid = uuid.uuid4()
        self.calibration = None
        self.calibration_path = None
        if calibration_path:
            self.calibration_path = calibration_path
            self.calibration = Calibration()
            self.calibration.load_from_file(path=calibration_path)
        # else: load_latest(calibration) -> own function used by all classes
        self.subject = None
        self.subject_path = None
        if subject_path:
            self.subject_path = subject_path
            self.subject = Subject().load_from_file(subject_path)
            self.subject.colospaces += [self]

        self.min_val = 0.00000000000001

        self.date = datetime.datetime.now()
        # TODO: get from current settings?
        self.bit_depth = bit_depth
        self.iso_slant = dict({})
        self.iso_slant["amplitude"] = 0
        self.iso_slant["phase"] = 0
        self.iso_slant["xdata"] = []
        self.iso_slant["ydata"] = []
        self.color_list = dict({})

        # dklc values
        self.chromaticity = chromaticity
        self.gray_level = gray_lavel
        self.unit = unit
        self.s_scale = s_scale

        # get from calibration
        if self.calibration:
            top = np.sum(self.rgb2lms([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), axis=0)
            self.lms_center = np.asarray((top + self.calibration.calibration_matrix[0]) / 2.)
        else:
            self.lms_center = np.asarray([0.5, 0.5, 0.5])

    def rgb2lms(self, rgb):
        """
        Conversion with gamma correction: lms = a_o + a * rgb**gamma.
        Uses calibration matrix.

        :param rgb: list/3-tuples/numpy array with single or multiple rgb values (0-1).
        :return: lms value as numpy array.
        """
        rgb = np.asarray(rgb)
        if rgb.ndim == 1:
            rgb = np.asarray([rgb])
        np.where(np.abs(rgb) < self.min_val, self.min_val, rgb)
        r, g, b = rgb.T

        cm = self.calibration.calibration_matrix
        v_p = np.asarray([np.power(r, cm[4][0]), np.power(g, cm[4][1]), np.power(b, cm[4][2])])
        lms = np.tile(cm[0], (len(rgb), 1)).T + np.dot(cm[1:4], v_p)

        return lms.T

    def lms2rgb(self, lms):
        """
        Conversion with gamma correction: x = (a_inv * (a_0 - lms))**(1/gamma_x).
        Uses inverse calibration matrix.

        :param lms: list/3-tuples/numpy array with single or multiple lms values (0-1).
        :return: rgb values as numpy array.
        """
        lms = np.asarray(lms)
        if lms.ndim == 1:
            lms = np.asarray([lms])

        icm = self.calibration.inv_calibration_matrix

        a = icm[1:4]
        v = lms - np.tile(icm[0], (len(lms), 1))
        va = v.dot(a.T).T
        r_g, g_g, b_g = np.where(np.abs(va) < self.min_val, self.min_val, va)

        rgb = np.asarray([np.power(r_g, icm[4][0]),
                          np.power(g_g, icm[4][1]),
                          np.power(b_g, icm[4][2])]).T

        return rgb

    @staticmethod
    def lms2dklc(lms):
        """
        Conversion of a lms value to dkl-similar coordinates,
        such that a gray value and color_angle can be given.
        If a subject is given, this also depends on its isoslant.

        :param lms: (list of) 3-tuple/numpy array with lms values (0-1).
        :return: dkl-like coordinates.
        """
        min_val = 0.00000000000001
        lms[lms == 0] = min_val
        l, m, s = lms

        return np.asarray((l+m, l-m, s))

    def dklc2lms(self, theta, gray=None, chromaticity=None, unit=None, s_scale=None):
        """
        Conversion of a dkl-similar value (gray/lum, theta) to a corresponding lms value.
        If a subject is given, this also depends on its iso-slant.

        :param theta: color angle(s).
        :param gray: luminance/gray value(s).
        :param chromaticity: Chromaticity.
        :param unit: unit for theta: rad or deg
        :param s_scale: Scaling factor for blue values.
        :return: lms values as numpy array.
        """
        # TODO: make sure that works with lists as well
        if gray is None:
            gray = self.lms_center
        if chromaticity is None:
            chromaticity = self.chromaticity
        if unit is None:
            unit = self.unit
        if s_scale is None:
            s_scale = self.s_scale

        if unit != 'rad':
            theta = 2. * theta * np.pi/360.

        phase = 0.
        if self.iso_slant["amplitude"] == 0.:
            amplitude = 0.
            print("WARNING: Amplitude of iso-slant is 0.\n"
                  "Make sure to measure subject's iso-slant with ColorSpace.measure_iso_slant.")
        else:
            amplitude = self.iso_slant["amplitude"]
            phase = self.iso_slant["phase"]

        # TODO: is that right?
        dlum = amplitude * np.sin(theta + phase)
        gray *= np.asarray([1.0 + dlum, 1.0 + dlum, 1.0])

        lm_ratio = 1.0 * gray[0] / gray[1]  # this ratio can be adjusted
        vec = np.asarray([
            1.0 + chromaticity * np.cos(theta) / (1.0 + lm_ratio),
            1.0 - chromaticity * np.cos(theta) / (1.0 + 1.0/lm_ratio),
            1.0 + s_scale * chromaticity * np.sin(theta)
        ])

        lms = gray * vec

        return lms

    def rgb2dklc(self, rgb):
        """
        Convert rgb value to dklc.
        :param rgb: (list of) 3-tuple/numpy array with rgb values [0, 1].
        :return: dklc values.
        """
        lms = self.lms2rgb(rgb)
        dklc = self.lms2dklc(lms)
        return dklc

    def dklc2rgb(self, theta, gray=None, chromaticity=None, unit=None, s_scale=None):
        """
        Conversion of a dkl-similar value (gray/lum, theta) to a corresponding rgb value.
        If a subject is given, this also depends on its iso-slant.

        :param theta: color angle(s).
        :param gray: luminance/gray value(s).
        :param chromaticity: Chromaticity.
        :param unit: unit for theta: rad or deg
        :param s_scale: Scaling factor for blue values.
        :return: rgb values as numpy array.
        """
        lms = self.dklc2lms(theta, gray, chromaticity, unit, s_scale)
        rgb = self.lms2rgb([lms])[0]
        return rgb

    @staticmethod
    def color2pp(xyz):
        """
        Convert rgb/lms values to psychopy xyz colorspace.

        :param xyz: list/3-tuples/numpy array with rgb/lms values [0, 1].
        :return: psychopy compatible values [-1, 1] as numpy array.
        """
        xyz = np.asarray(xyz)
        if xyz.ndim == 1:
            xyz = np.asarray([xyz])
        xyz_pp = 2. * xyz - np.ones((len(xyz), len(xyz[0])))
        return xyz_pp

    @staticmethod
    def pp2color(xyz_pp):
        """
        Convert psychopy rgb/lms values to psychopy xyz colorspace.

        :param xyz_pp: list/3-tuple/numpy array with psychopy rgb/lms values [-1, 1].
        :return: rgb/lms values [0, 1] as numpy array.
        """
        xyz_pp = np.asarray(xyz_pp)
        if xyz_pp.ndim == 1:
            xyz_pp = np.asarray([xyz_pp])
        xyz = (xyz_pp + np.ones((len(xyz_pp), len(xyz_pp[0])))) * 0.5
        return xyz

    def rgb2rgb255(self, rgb):
        """
        Convert rgb values to rgb255.

        :param rgb: (list of) 3-tuple/numpy array with rgb values [0, 1].
        :return: rgb255 values [0, 255] as numpy array.
        """

        if self.bit_depth != 8:
            print("WARNING: The current bit depth of the monitor is not 8 bit."
                  "Conversion to rgb255 format possibly leads to differences in color.")
        rgb = np.asarray(rgb)
        if rgb.ndim == 1:
            rgb = np.asarray([rgb])
        rgb255 = (rgb * 255 + 0.5 * np.ones((len(rgb), len(rgb[0])))).astype(int)
        return rgb255

    def rgb2552rgb(self, rgb255):
        """
        Convert rgb255 values to rgb.

        :param rgb255: (list of) 3-tuple/numpy array with rgb255 values [0, 255].
        :return: rgb values [0, 1] as numpy array.
        """

        if self.bit_depth != 8:
            print("WARNING: The current bit depth of the monitor is not 8 bit."
                  "Consider using float rgb values in range [0, 1].")
        rgb255 = np.asarray(rgb255).astype(float)
        if rgb255.ndim == 1:
            rgb255 = np.asarray([rgb255])
        rgb = rgb255/255.
        return rgb

    def rgb2rgb1023(self, rgb):
        """
        Convert rgb values to rgb1023.

        :param rgb: (list of) 3-tuple/numpy array with rgb values [0, 1].
        :return: rgb255 values [0, 1023] as numpy array.
        """

        if self.bit_depth != 10:
            print("WARNING: The current bit depth of the monitor is not 10 bit."
                  "Conversion to rgb1023 format possibly leads to differences in color.")
        rgb = np.asarray(rgb)
        if rgb.ndim == 1:
            rgb = np.asarray([rgb])
        rgb1023 = (rgb * 1023 + 0.5*np.ones((len(rgb), len(rgb[0])))).astype(int)
        return rgb1023

    def rgb10232rgb(self, rgb1023):
        """
        Convert rgb1023 values to rgb.

        :param rgb1023: (list of) 3-tuple/numpy array with rgb1023 values [0, 1023].
        :return: rgb values [0, 1] as numpy array.
        """

        if self.bit_depth != 10:
            print("WARNING: The current bit depth of the monitor is not 8 bit."
                  "Consider using float rgb values in range [0, 1].")
        rgb1023 = np.asarray(rgb1023).astype(float)
        if rgb1023.ndim == 1:
            rgb1023 = np.asarray([rgb1023])
        rgb = rgb1023/1023.
        return rgb

    def dklc_gray(self, dlum, lms_gray=None):
        """
        Get a dklc gray value.
        :param lms_gray: Gray reference value. If None, lms_center is used.
        :param dlum: Change in luminance.
        :return: New gray value in lms-coordinates.
        """

        if lms_gray is None:
            lms_gray = self.lms_center

        return lms_gray * (np.ones(3) + np.asarray([dlum/2., dlum/2., 0.]))

    def measure_iso_slant(self, gray_level=None, num_fit_points=8, repeats=2, lim=0.1,
                          step_size=0.001, refresh=60):
        """
        Run luminance fitting experiment and fit a sine-function to
        get the iso-slant for iso-luminance plane.
        Depends on the current calibration.

        :param gray_level: Luminance value.
        :param num_fit_points: Number of angles for fitting.
        :param repeats: Number of trials for each point.
        :param step_size: Change in luminance with each key event.
        :param lim: Limit for mouse range.
        :param refresh: 1/refresh is frame length.
        """

        if gray_level is None:
            gray_level = self.gray_level
        rgb_gray = [[gray_level, gray_level, gray_level]]
        lms_gray = self.rgb2lms(rgb_gray)[0]

        response = np.zeros((2, repeats * num_fit_points))
        stimulus = np.linspace(0, 2 * np.pi, num_fit_points, endpoint=False)
        randstim = np.random.permutation(np.repeat(stimulus, repeats))

        win_h = 800
        win_w = 600
        # each stimulus lasts 4 frames; each frame last for 1/refresh second
        keep = 2
        freq = refresh / keep

        for idx, theta in enumerate(randstim):
            win = visual.Window([win_h, win_w], monitor="eDc-1")

            mouse = event.Mouse()

            bg = visual.Rect(win, pos=[0, 0], width=1., height=1.)
            bg.setColor(self.color2pp(rgb_gray)[0], "rgb")
            rect = visual.Rect(win, pos=[0, 0], width=0.35, height=0.5)
            color = self.color2pp([self.dklc2rgb(theta, gray=lms_gray)])[0]
            rect.setColor(color, "rgb")
            text = visual.TextStim(win, pos=[-0.7, 0.95], height=0.03,
                                   text=str(idx + 1) + ' of ' + str(len(randstim)) +
                                   ' stimuli at ' + str(freq) + 'Hz')

            dlum = 0.
            frameN = 0
            curr_color = color
            pos, _ = mouse.getPos()

            while True:
                if frameN % (2 * keep) < keep:

                    # get mouse position.
                    x, _ = mouse.getPos()
                    if x != pos:
                        dlum = lim * x
                        pos = x
                        refgray = self.dklc_gray(dlum, lms_gray=lms_gray)
                        color = self.color2pp([self.dklc2rgb(theta, gray=refgray)])[0]
                        if len(color[color > 1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color

                    rect.setColor(curr_color, "rgb")
                    rect.draw()

                    if event.getKeys('right'):
                        # dlum = gray_level * (1 + dlum)
                        refgray = self.dklc_gray(dlum + step_size, lms_gray=lms_gray)
                        color = self.color2pp([self.dklc2rgb(theta, gray=refgray)])[0]
                        if len(color[color > 1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color
                            dlum += step_size

                    if event.getKeys('left'):
                        refgray = self.dklc_gray(dlum - step_size, lms_gray=lms_gray)
                        color = self.color2pp([self.dklc2rgb(theta, gray=refgray)])[0]
                        if len(color[color < -1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color
                            dlum -= step_size

                    if event.getKeys('space'):
                        win.close()
                        break

                text.draw()
                win.flip()

                frameN += 1

            response[0][idx] = theta
            response[1][idx] = dlum

        stim, res = response
        params, _ = optimize.curve_fit(sine_fitter, stim, res)

        self.iso_slant["amplitude"] = params[0]
        self.iso_slant["phase"] = params[1]
        self.iso_slant["xdata"] = stim
        self.iso_slant["ydata"] = res

    def create_color_list(self, hue_res=0.2, gray_level=None):
        """
        Generate colors that are realizable in a N-bit display and save them in a color list.
        :param hue_res: Resolution of hue angles, i.e. hue angle bins. Given in DEG!
        :param gray_level: Luminance value.
        """
        if gray_level is None:
            gray_level = self.gray_level

        theta = np.linspace(0, 360 - hue_res, int(360 / hue_res))
        conv_th = [self.dklc2rgb(theta=th, gray=gray_level, unit="deg") for th in theta]
        rgb = [c_th for c_th in conv_th]
        # resolution of N bit in [0, 1] scale
        rgb_res = 1. / np.power(2., self.bit_depth)
        sel_rgb = []
        sel_theta = []
        rgb_len = len(rgb)
        it = iter(list(range(0, len(rgb))))

        for idx in it:
            sel_rgb.append(rgb[idx])
            sel_theta.append(theta[idx])
            # prevent multiple entries
            if abs(rgb[idx][1] - rgb[(idx + 1) % rgb_len][1]) <= rgb_res:
                next(it)
            if abs(rgb[idx][1] - rgb[(idx + 2) % rgb_len][1]) <= rgb_res:
                next(it)
                next(it)

        # add list to subject and colorspace
        self.color_list[hue_res] = dict({})
        self.color_list[hue_res]["hue_angles"] = sel_theta
        self.color_list[hue_res]["rgb"] = sel_rgb

    def plot_iso_slant(self):
        """
        Run input and fit a sine-function to get the iso-slant for iso-luminance plane.
        Depends on the current calibration.
        """

        x = np.arange(0., 2.*np.pi, 0.01*np.pi)
        a = self.iso_slant["amplitude"]
        ph = self.iso_slant["phase"]

        xdata = self.iso_slant["xdata"]
        ydata = self.iso_slant["ydata"]

        f = a*np.sin(x + ph)
        pl.plot(x, f, label="fit")
        pl.plot(xdata, ydata, 'x', label="data")
        pl.grid()
        pl.xlabel("Phase")
        pl.ylabel("Delta Luminance")
        pl.legend()
        pl.tight_layout()
        pl.show()

    def show_color_circle(self, num_col=16, gray_level=None):
        """
        Show color circle.
        :param num_col: number of colors to be shown.
        :param gray_level: gray level.
        """

        if gray_level is None:
            gray_level = self.gray_level

        phis = np.linspace(0, 2 * np.pi, num_col, endpoint=False)
        m_rgb = []
        for phi in phis:
            m_rgb.append(self.dklc2rgb(theta=phi))

        bg_color = self.color2pp([[gray_level, gray_level, gray_level]])[0]

        win = visual.Window(size=[800, 600], colorSpace="rgb", color=bg_color, allowGUI=True,
                            bit_depth=self.bit_depth)

        rect_size = 0.4 * win.size[0] * 2 / num_col
        radius = 0.2 * win.size[0]
        alphas = np.linspace(0, 360, num_col, endpoint=False)

        rect = visual.Rect(win=win,
                           units="pix",
                           width=int(rect_size), height=int(rect_size))
        for i_rect in range(num_col):
            rect.fillColorSpace = "rgb"
            rect.lineColorSpace = "rgb"
            rect.fillColor = m_rgb[i_rect]
            rect.lineColor = m_rgb[i_rect]
            rect.pos = misc.pol2cart(alphas[i_rect], radius)
            rect.draw()

        win.flip()

        event.waitKeys()
        win.close()

    def save_to_file(self, path=None, directory=None):
        """
        Save object data to file.
        :param path: location of file.
        :param directory: directory, if file name should be filled automatically.
        """
        dt = {}
        dt.update(vars(self))
        del dt["calibration"]
        del dt["subject"]
        dt["uuid"] = str(self.uuid)
        dt["date"] = str(self.date)
        dt["lms_center"] = self.lms_center.tolist()
        iso_slant = self.iso_slant
        iso_slant["xdata"] = np.asarray(iso_slant["xdata"]).tolist()
        iso_slant["ydata"] = np.asarray(iso_slant["ydata"]).tolist()
        iso_slant = json.dumps(iso_slant)
        dt["iso_slant"] = iso_slant
        for key, val in self.color_list.items():
            self.color_list[key]["hue_angles"] = np.asarray(val["hue_angles"]).tolist()
            self.color_list[key]["rgb"] = np.asarray(val["rgb"]).tolist()
        dt["color_list"] = json.dumps(self.color_list)

        if not path:
            path = "colorspace_{}.json".format(self.date)
        if directory:
            path = os.path.join(directory, path)
        save_dir, save_file = os.path.split(path)
        if save_dir and not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if ".json" not in save_file:
            path = path + ".json"
        json.dump(dt, codecs.open(path, 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True, indent=4)

        print("Successfully saved colorspace to file {}".format(path))

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

        if self.calibration_path:
            self.calibration = Calibration()
            self.calibration.load_from_file(self.calibration_path)
        if self.subject_path:
            self.subject = Subject()
            self.subject.load_from_file(self.subject_path)

        self.iso_slant = json.loads(self.iso_slant)
        self.color_list = json.loads(self.color_list)

        print("Successfully loaded colorspace from file {}".format(path))
