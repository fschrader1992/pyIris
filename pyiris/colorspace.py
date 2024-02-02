# -*- coding: utf-8
"""
This class contains all properties/methods related to color spaces.
Latest version: 2.0.0.
"""

import os
import datetime
import uuid
import json
import itertools
import ruamel.yaml
from operator import itemgetter
from psychopy import event, misc, visual
from scipy import optimize

import numpy as np
import matplotlib.pyplot as plt

from .monitor import Monitor
from .subject import Subject
from .functions import dump_file, float2hex, sine_fitter
try:
    from .calibration import Calibration
except ImportError:
    pass


class ColorSpace:
    """
    Class Colorspace
    """

    def __init__(self, calibration_path=None, subject_path=None, bit_depth=10, saturation=0.16,
                 gray_level=0.66, unit="rad", s_scale=2.6):

        self.uuid = uuid.uuid4()
        self.calibration_path = None
        self.calibration = None
        self.monitor = None
        self.bit_depth = bit_depth
        if calibration_path:
            self.calibration_path = calibration_path
            self.calibration = Calibration()
            self.calibration.load_from_file(path=calibration_path)
            self.monitor = Monitor(settings_path=self.calibration.monitor_settings_path)
            self.bit_depth = self.monitor.bpc

        # else: load_latest(calibration) -> own function used by all classes
        self.subject_path = subject_path if subject_path else None
        self.subject = None
        if subject_path:
            self.subject = Subject()
            self.subject.load_from_file(subject_path)

        self.min_val = 0.00000000000001

        self.date = datetime.datetime.now()
        self.iso_slant = dict({})
        self.iso_slant["amplitude"] = 0
        self.iso_slant["phase"] = 0
        self.iso_slant["xdata"] = []
        self.iso_slant["ydata"] = []
        self.iso_slant_sep = dict({})
        self.color_list = dict({})

        # cnop values
        self.saturation = saturation
        self.gray_level = gray_level
        self.unit = unit
        self.s_scale = s_scale

        self.checkerboard = None

        self.op_mode = False

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

    def lms2dklc(self, lms):
        """
        Wrapper for old nomenclature.
        """
        return self.lms2cnop(lms)

    def lms2cnop(self, lms):
        """
        Conversion of a lms value to dkl-similar coordinates,
        such that a gray value and color_angle can be given.
        If a subject is given, this also depends on its isoslant.

        :param lms: list/3-tuples/numpy array with single or multiple lms values (0-1).
        :return: dkl-like coordinates.
        """
        lms[lms == 0] = self.min_val
        if lms.ndim == 1:
            lms = np.asarray([lms])
        l, m, s = lms.T

        r = l+m
        a = l-m
        phi = np.arccos(a/r)
        return phi

    def dklc2lms(self, phi, gray_level=None, saturation=None, unit=None, s_scale=None):
        """
        Wrapper for old nomenclature.
        """
        return self.cnop2lms(phi, gray_level=gray_level, saturation=saturation, unit=unit, s_scale=s_scale)

    def cnop2lms(self, phi, gray_level=None, saturation=None, unit=None, s_scale=None):
        """
        Conversion of a dkl-similar value (gray/lum, phi) to a corresponding lms value.
        If a subject is given, this also depends on its iso-slant.

        :param phi: color angle(s) as list/numpy array.
        :param gray_level: luminance/gray value(s).
        :param saturation: saturation.
        :param unit: unit for phi: rad or deg
        :param s_scale: Scaling factor for blue values.
        :return: lms values as numpy array.
        """

        phi = np.asarray(phi)
        if phi.ndim == 0:
            phi = np.asarray([phi])
        phi_len = len(phi)
        if saturation is None:
            saturation = self.saturation
        if phi.ndim == 0:
            saturation = saturation * np.ones(phi_len)
        if gray_level is None:
            gray_level = np.array([self.gray_level])
        gray_level = np.asarray(gray_level)
        if gray_level.ndim == 0:
            gray_level = np.asarray([gray_level])
        gray = np.tile(gray_level, (3, 1)).T
        if unit is None:
            unit = self.unit
        if s_scale is None:
            s_scale = self.s_scale
        s_scale = s_scale * np.ones(phi_len)
        if unit != 'rad':
            phi *= np.pi/180.

        amplitude = 0.
        phase = 0.
        offset = 0.
        chrom_0 = self.saturation
        if self.iso_slant["amplitude"] == 0.:
            if not self.op_mode:
                print("WARNING: Amplitude of iso-slant is 0.\n"
                      "Make sure to measure subject's iso-slant with ColorSpace.measure_iso_slant.")
        else:
            amplitude = self.iso_slant["amplitude"]
            phase = self.iso_slant["phase"]
            offset = self.iso_slant["offset"]
            chrom_0 = self.iso_slant["saturation"]

        gray_level = np.repeat(gray_level, phi_len, axis=0)
        phase = phase * np.ones(phi_len)
        phi_lum = phi + phase

        gray_level = [gray_level + saturation/chrom_0 * amplitude * np.sin(phi_lum) + offset]
        gray = self.rgb2lms(np.repeat(gray_level, 3, axis=0).T)
        gray[gray == 0] = self.min_val

        # this ratio can be adjusted
        lm_ratio = 1.0 * gray.T[0] / gray.T[1]

        vec = np.asarray([
            1.0 + saturation * np.cos(phi) / (1.0 + lm_ratio),
            1.0 - saturation * np.cos(phi) / (1.0 + 1.0/lm_ratio),
            1.0 + s_scale * saturation * np.sin(phi)
        ]).T

        lms = gray * vec

        return lms

    def rgb2dklc(self, rgb):
        """
        Wrapper for old nomenclature.
        """
        return self.rgb2cnop(rgb)

    def rgb2cnop(self, rgb):
        """
        Convert rgb value to cnop.
        :param rgb: (list of) 3-tuple/numpy array with rgb values [0, 1].
        :return: cnop values.
        """
        lms = self.lms2rgb(rgb)
        cnop = self.lms2cnop(lms)
        return cnop

    def dklc2rgb(self, phi, gray_level=None, saturation=None, unit=None, s_scale=None):
        """
        Wrapper for old nomenclature.
        """
        return self.cnop2rgb(phi, gray_level=gray_level, saturation=saturation, unit=unit, s_scale=s_scale)

    def cnop2rgb(self, phi, gray_level=None, saturation=None, unit=None, s_scale=None):
        """
        Conversion of a dkl-similar value (gray/lum, phi) to a corresponding rgb value.
        If a subject is given, this also depends on its iso-slant.

        :param phi: color angle(s).
        :param gray_level: luminance/gray value(s).
        :param saturation: saturation.
        :param unit: unit for phi: rad or deg
        :param s_scale: Scaling factor for blue values.
        :return: rgb values as numpy array.
        """
        lms = self.cnop2lms(phi, gray_level, saturation, unit, s_scale)
        rgb = self.lms2rgb([lms])[0]
        return rgb

    def color2pp(self, xyz):
        """
        Wrapper for old nomenclature.
        """
        return self.rgb2pp(xyz)

    @staticmethod
    def rgb2pp(xyz):
        """
        Convert rgb/lms values to psychopy xyz colorspace.

        :param xyz: list/3-tuples/numpy array with rgb values in interval [0, 1].
        :return: psychopy compatible values [-1, 1] as numpy array.
        """
        xyz = np.asarray(xyz)
        if xyz.ndim == 1:
            xyz = np.asarray([xyz])
        xyz_pp = 2. * xyz - np.ones((len(xyz), len(xyz[0])))
        return xyz_pp

    def pp2color(self, xyz):
        """
        Wrapper for old nomenclature.
        """
        return self.pp2rgb(xyz)

    @staticmethod
    def pp2rgb(xyz_pp):
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

    @staticmethod
    def rgb2552hex(rgb255, cross=True):
        """
        Convert rgb255 values to hex.
        :param rgb255: (list of) 3-tuple/numpy array with rgb255 values [0, 255].
        :param cross: If True, indicator "#" is added.
        :return: hex values as numpy array.
        """

        hex_arr = []
        start = "#" if cross else ""
        thf = np.vectorize(float2hex)
        rgb255 = thf(rgb255)
        for r in rgb255:
            hex_arr += [start + "".join(r)]
        hex_arr = np.asarray(hex_arr)
        return hex_arr

    def rgb2hex(self, rgb, cross=True):
        """
        Convert rgb values to hex.
        :param rgb: (list of) 3-tuple/numpy array with rgb values [0, 1].
        :param cross: If True, indicator "#" is added.
        :return: hex values as numpy array.
        """

        rgb255 = self.rgb2rgb255(rgb)
        hex_arr = self.rgb2552hex(rgb255, cross=cross)
        return hex_arr

    @staticmethod
    def hex2rgb255(hex_arr):
        """
        Convert hex values to rgb255.

        :param hex_arr: (3- or 6-digit) hex values (with/-out "#") as numpy array.
        :return: (list of) 3-tuple/numpy array with rgb255 values [0, 255].
        """

        hs2ha = lambda t, ti, lti: int(t.lstrip("#")[ti:ti+lti] if lti == 2 else
                                       t.lstrip("#")[ti:ti+lti] + t.lstrip("#")[ti:ti+lti], 16)
        splitter = lambda t: tuple([hs2ha(t, 0, 2), hs2ha(t, 2, 2), hs2ha(t, 4, 2)])\
            if len(t) == 6 or len(t) == 7 else tuple([hs2ha(t, 0, 1), hs2ha(t, 1, 1), hs2ha(t, 2, 1)])
        rgb255 = np.asarray(np.vectorize(splitter)(hex_arr)).T
        return rgb255

    def hex2rgb(self, hex_arr):
        """
        Convert hex values to rgb.

        :param hex_arr: (3- or 6-digit) hex values (with/-out "#") as numpy array.
        :return: (list of) 3-tuple/numpy array with rgb255 values [0, 1].
        """

        r2 = self.hex2rgb255(hex_arr)
        rgb = self.rgb2552rgb(r2)
        return rgb

    def measure_iso_slant(self, gray_level=None, num_fit_points=8, repeats=10, lim=0.1,
                          step_size=0.001, refresh=None, field='norm'):
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
        :param field: Whether to test with normal stimulus (8x8deg),
                      only "center" region (3x3) or the "periphery/ecc"entricity
                      (12x12, with 8x8 cutout)
        """

        self.op_mode = True
        if gray_level is None:
            gray_level = self.gray_level
        if refresh is None:
            refresh = self.monitor.refresh

        response = np.zeros((2, repeats * num_fit_points))
        stimulus = np.linspace(0., 2. * np.pi, num_fit_points, endpoint=False)
        randstim = np.random.permutation(np.repeat(stimulus, repeats))

        #  get correct frames, frequency should be between 10-20 Hz
        # number of frames with stimulus (half of the frames for the frequency)
        keep = int(refresh/(2.*15.))
        freq = refresh / keep / 2.

        win = visual.window.Window(
            size=[self.monitor.currentCalib['sizePix'][0], self.monitor.currentCalib['sizePix'][1]],
            monitor=self.monitor, fullscr=True, colorSpace='rgb',
        )

        # set background gray level
        win.colorSpace = "rgb"
        win_color = self.rgb2pp(np.array([gray_level, gray_level, gray_level]))[0]
        win.color = win_color

        mouse = event.Mouse()

        info = visual.TextStim(win, pos=[0, 12], height=0.5, units='deg')
        info.autoDraw = True

        fix = visual.TextStim(win, text="+", pos=[0, 0], height=0.6, color='black', units='deg')

        rect2 = None
        if 'ecc' in field or 'peri' in field:
            rect = visual.Rect(win, pos=[0, 0], width=24, height=24, units='deg')
            rect2 = visual.Rect(win, pos=[0, 0], width=16, height=16, units='deg')
            rect2.setColor(win_color, "rgb")
        elif field == 'center':
            rect = visual.Rect(win, pos=[0, 0], width=4, height=4, units='deg')
        elif field == 'fovea':
            rect = visual.Rect(win, pos=[0, 0], width=1.5, height=1.5, units='deg')
        else:
            rect = visual.Rect(win, pos=[0, 0], width=8, height=8, units='deg')

        for idx, phi in enumerate(randstim):
            info.text = str(idx + 1) + ' of ' + str(len(randstim)) +\
                        ' stimuli at ' + str(freq) + 'Hz'

            color = self.rgb2pp(self.cnop2rgb(phi, gray_level=gray_level))[0]
            rect.setColor(color, "rgb")

            d_gray = 0.
            i_frame = 0
            curr_color = color
            mouse.setPos(0.)
            pos, _ = mouse.getPos()

            while True:
                if i_frame % (2 * keep) < keep:
                    # get mouse position.
                    x, _ = mouse.getPos()
                    if x != pos:
                        d_gray = lim * x
                        pos = x
                        ref_gray_level = gray_level + d_gray
                        color = self.rgb2pp(self.cnop2rgb(phi, gray_level=ref_gray_level))[0]
                        if len(color[color > 1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color

                    rect.setColor(curr_color, "rgb")
                    rect.draw()
                    if 'ecc' in field or 'peri' in field:
                        rect2.draw()

                    if event.getKeys('right'):
                        ref_gray_level = gray_level + np.ones(3) * (d_gray + step_size)
                        color = self.rgb2pp(self.cnop2rgb(phi, gray_level=ref_gray_level))[0]
                        if len(color[color > 1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color
                            d_gray += step_size

                    if event.getKeys('left'):
                        ref_gray_level = gray_level + np.ones(3) * (d_gray - step_size)
                        color = self.rgb2pp(self.cnop2rgb(phi, gray_level=ref_gray_level))[0]
                        if len(color[color < -1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color
                            d_gray -= step_size

                    if event.getKeys('space'):
                        break

                if field != 'norm':
                    fix.draw()
                win.flip()

                i_frame += 1

            response[0][idx] = phi
            response[1][idx] = d_gray

        win.close()
        stim, res = response
        params, _ = optimize.curve_fit(sine_fitter, stim, res)

        self.iso_slant["amplitude"] = params[0]
        self.iso_slant["phase"] = params[1]
        self.iso_slant["offset"] = params[2]
        self.iso_slant["xdata"] = stim
        self.iso_slant["ydata"] = res
        self.iso_slant["saturation"] = self.saturation
        self.iso_slant["gray_level"] = gray_level
        self.op_mode = False

        return True

    def measure_iso_slant_sep(self, gray_level=None, num_fit_points=8, repeats=6, lim=0.1,
                              step_size=0.001, refresh=None, posxs=None, posys=None,
                              pos_labels=None, size=6):
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
        :param posxs: x-positions of test patches. If not given
                      four test patches will be placed at +/-3.5 deg, also for the height.
        :param posys: y-position of test patches, should have same length as posxs.
        :param pos_labels: Labels for the respective patches for later analysis.
                      If not given "up_left", "up_right", "down_left", "down_right" are used correspondingly.
        :param size: Size of test patches.
        """

        self.op_mode = True
        if gray_level is None:
            gray_level = self.gray_level
        if refresh is None:
            refresh = self.monitor.refresh

        # response = np.zeros((2, repeats * num_fit_points))
        response = dict({})
        stimulus = np.linspace(0., 2. * np.pi, num_fit_points, endpoint=False)
        # randstim = np.random.permutation(np.repeat(stimulus, repeats))

        #  get correct frames, frequency should be between 10-20 Hz
        # number of frames with stimulus (half of the frames for the frequency)
        keep = int(refresh/(2.*15.))
        freq = refresh / keep / 2.

        # win = visual.Window([self.monitor.currentCalib['sizePix'][0],
        #                      self.monitor.currentCalib['sizePix'][1]],
        #                     monitor=self.monitor.name, fullscr=True, units='deg')
        win = visual.window.Window(
            size=[self.monitor.currentCalib['sizePix'][0], self.monitor.currentCalib['sizePix'][1]],
            monitor=self.monitor, fullscr=True, colorSpace='rgb',
        )

        # set background gray level
        win.colorSpace = "rgb"
        win.color = self.rgb2pp(np.array([gray_level, gray_level, gray_level]))[0]

        mouse = event.Mouse()

        info = visual.TextStim(win, pos=[0, 12], height=0.5, units='deg')
        info.autoDraw = True

        if posxs is None:
            pos_labels = ['up_left', 'up_right', 'down_left', 'down_right']
            posxs = [-3.5, 3.5, -3.5, 3.5]
            posys = [3.5, 3.5, -3.5, -3.5]
        # poss = [[-3.5, 3.5], [3.5, 3.5], [-3.5, -3.5], [3.5, -3.5]]
        fix = visual.TextStim(win, text="+", pos=[0, 0], height=0.6, color='black', units='deg')
        fix.autoDraw = True

        for pos_label in pos_labels:
            response[pos_label] = dict({})
            response[pos_label]['x'] = []
            response[pos_label]['y'] = []

        self.iso_slant_sep = dict({})

        # print(poss)
        # print('---')
        stim_r = np.tile(stimulus, (repeats * len(pos_labels),))
        posxs_r = np.repeat(posxs, repeats * num_fit_points)
        posys_r = np.repeat(posys, repeats * num_fit_points)
        # print(poss_r)
        pos_labels_r = np.repeat(pos_labels, repeats * num_fit_points)
        num_r = np.random.permutation(np.arange(len(stim_r)))
        stim_r = stim_r[num_r]
        posxs_r = posxs_r[num_r]
        posys_r = posys_r[num_r]
        pos_labels_r = pos_labels_r[num_r]

        for idx, (phi, pos_label, posx, posy) in enumerate(zip(stim_r, pos_labels_r, posxs_r, posys_r)):

            pos = (posx, posy)
            # randstim = np.random.permutation(np.repeat(stimulus, repeats))

            rect = visual.Rect(win, pos=pos, width=size, height=size, units='deg')
            # rect = visual.Rect(win, pos=[0.06, 0.06], width=0.06, height=0.08)

            # for idx, phi in enumerate(randstim):
            info.text = str(idx + 1) + ' of ' + str(len(stim_r)) +\
                        ' stimuli at ' + str(freq) + 'Hz'

            color = self.rgb2pp(self.cnop2rgb(phi, gray_level=gray_level))[0]
            rect.setColor(color, "rgb")

            d_gray = 0.
            i_frame = 0
            curr_color = color
            mouse.setPos(0.)
            pos, _ = mouse.getPos()

            while True:
                if i_frame % (2 * keep) < keep:
                    # get mouse position.
                    x, _ = mouse.getPos()
                    if x != pos:
                        d_gray = lim * x
                        pos = x
                        ref_gray_level = gray_level + d_gray
                        color = self.rgb2pp(self.cnop2rgb(phi, gray_level=ref_gray_level))[0]
                        if len(color[color > 1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color

                    rect.setColor(curr_color, "rgb")
                    rect.draw()

                    if event.getKeys('right'):
                        ref_gray_level = gray_level + np.ones(3) * (d_gray + step_size)
                        color = self.rgb2pp(self.cnop2rgb(phi, gray_level=ref_gray_level))[0]
                        if len(color[color > 1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color
                            d_gray += step_size

                    if event.getKeys('left'):
                        ref_gray_level = gray_level + np.ones(3) * (d_gray - step_size)
                        color = self.rgb2pp(self.cnop2rgb(phi, gray_level=ref_gray_level))[0]
                        if len(color[color < -1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color
                            d_gray -= step_size

                    if event.getKeys('space'):
                        break

                win.flip()

                i_frame += 1

            # response[0][idx] = phi
            response[pos_label]['x'] += [phi]
            response[pos_label]['y'] += [d_gray]

        for pos_label in pos_labels:
            stim, res = response[pos_label]['x'], response[pos_label]['y']
            params, _ = optimize.curve_fit(sine_fitter, stim, res)

            self.iso_slant_sep[pos_label] = dict({})
            self.iso_slant_sep[pos_label]["amplitude"] = params[0]
            self.iso_slant_sep[pos_label]["phase"] = params[1]
            self.iso_slant_sep[pos_label]["offset"] = params[2]
            self.iso_slant_sep[pos_label]["xdata"] = stim
            self.iso_slant_sep[pos_label]["ydata"] = res
            self.iso_slant_sep[pos_label]["saturation"] = self.saturation
            self.iso_slant_sep[pos_label]["gray_level"] = gray_level
        win.close()

        self.op_mode = False

        return True

    def create_color_list(self, axes="cnop_hues",
                          hue_angle=None, hue_res=0.2, min_hue=0., max_hue=360.0001,
                          saturation=None, sat_res=0.01, min_sat=0.0, max_sat=0.3001,
                          gray_level=None, lum_res=0.01, min_lum=0.0, max_lum=1.0001):
        """
        Generate colors that are realizable in a N-bit display and save them in a color list.
        :param axes: Tuple containing the definitions of dimensions along which the color list should be built.
               Possible axes are "cnop_hues" (hue angle), "cnop_sat" (saturation) and "cnop_lum" (luminance).
               If only one value is given, it can be set as simple string.
               If only two are given, the respective entries/keys in the color list's "cnop" field will be ordered
               in the way given.
               If all three dimensions are listed, the respective entries/keys in the color list's "cnop" field
               will always bein the order ["cnop_hues", "cnop_sat", "cnop_lum"].
               Default is "cnop_hues".
        :param hue_angle: Optional hue angle, used when only "cnop_sat" or cnop_lum" is given. Default is None.
        :param hue_res: Resolution of hue angles, i.e. hue angle bins. Given in DEG hue angle!
        :param min_hue: Minimum hue angle. Default is 0.0.
        :param max_hue: Maximum hue angle up to which values are generated. Default is 360.0001.
        :param saturation: Base saturation,used for "cnop_hues" and "cnop_lum".
               If not given, class attribute settings are used.
        :param sat_res: Resolution for saturation steps. Default is 0.01.
        :param min_sat: Minimum saturation value. Default is 0.0.
        :param max_sat: Maximum saturation up to which values are generated. Default is 0.3001.
        :param gray_level: Luminance value with "cnop_hues" or "cnop_sat".
               If not given, class attribute settings are used.
        :param lum_res: Resolution for luminance steps. Default is 0.01.
        :param min_lum: Minimum luminance value. Default is 0.0.
        :param max_lum: Maximum luminance up to which values are generated. Default is 1.0001.
        :return: True.
        """

        self.op_mode = True
        if gray_level is None:
            gray_level = self.gray_level
        if saturation is None:
            saturation = self.saturation
        if min_hue > max_hue:
            max_hue += 360.

        # list containing values in cnop colorspace, can be scalars or arrays/lists for each dimension
        cnop_vals = None
        # array containing RGB values for each list entry
        rgbs = None

        if isinstance(axes, str) or len(axes) == 1:
            if "cnop_hues" in axes:
                cnop_vals = np.arange(min_hue, max_hue, hue_res)
                rgbs = np.array([
                    self.cnop2rgb(
                        phi=phi, saturation=saturation, gray_level=gray_level, unit="deg"
                    )[0].tolist() for phi in cnop_vals
                ])
            elif "cnop_sat" in axes:
                cnop_vals = np.arange(min_sat, max_sat, sat_res)
                hue_angles = hue_angle * np.ones(len(cnop_vals))
                hue_angles[np.argwhere(cnop_vals < 0.)] += 180.
                hue_angles %= 360.
                rgbs = np.array([
                    self.cnop2rgb(
                        phi=phi, saturation=sat, gray_level=gray_level, unit="deg"
                    )[0].tolist() for phi, sat in zip(hue_angles, np.abs(cnop_vals))
                ])
            elif "cnop_lum" in axes:
                cnop_vals = np.arange(min_lum, max_lum, lum_res)
                rgbs = np.array([
                    self.cnop2rgb(
                        phi=hue_angle, saturation=saturation, gray_level=gray_lev, unit="deg"
                    )[0].tolist() for gray_lev in cnop_vals
                ])
            else:
                raise ValueError('Unknown type "{}" for color list "axes" parameter.'.format(axes[0]))
        elif not isinstance(axes, str) and len(axes) == 2:
            if axes == ("cnop_hues", "cnop_sat") or axes == ("cnop_sat", "cnop_hues"):
                hues = np.arange(min_hue, max_hue, hue_res).tolist()
                sat = np.arange(min_sat, max_sat, sat_res).tolist()
                if axes == ("cnop_hues", "cnop_sat"):
                    cnop_vals = np.array(list(itertools.product(hues, sat)))
                    rgbs = np.array([
                        self.cnop2rgb(
                            phi=phi, saturation=sat, gray_level=gray_level, unit="deg"
                        )[0].tolist() for (phi, sat) in cnop_vals
                    ])
                else:
                    cnop_vals = np.array(list(itertools.product(sat, hues)))
                    rgbs = np.array([
                        self.cnop2rgb(
                            phi=phi, saturation=sat, gray_level=gray_level, unit="deg"
                        )[0].tolist() for (sat, phi) in cnop_vals
                    ])
            elif axes == ("cnop_hues", "cnop_lum") or axes == ("cnop_lum", "cnop_hues"):
                hues = np.arange(min_hue, max_hue, hue_res).tolist()
                lum = np.arange(min_lum, max_lum, lum_res).tolist()
                if axes == ("cnop_hues", "cnop_lum"):
                    cnop_vals = np.array(list(itertools.product(hues, lum)))
                    rgbs = np.array([
                        self.cnop2rgb(
                            phi=phi, saturation=saturation, gray_level=lum, unit="deg"
                        )[0].tolist() for (phi, lum) in cnop_vals
                    ])
                else:
                    cnop_vals = np.array(list(itertools.product(lum, hues)))
                    rgbs = np.array([
                        self.cnop2rgb(
                            phi=phi, saturation=saturation, gray_level=lum, unit="deg"
                        )[0].tolist() for (lum, phi) in cnop_vals
                    ])
            elif axes == ("cnop_sat", "cnop_lum") or axes == ("cnop_lum", "cnop_sat"):
                sats = np.arange(min_sat, max_sat, sat_res).tolist()
                lum = np.arange(min_lum, max_lum, lum_res).tolist()
                if axes == ("cnop_sat", "cnop_lum"):
                    cnop_vals = np.array(list(itertools.product(sats, lum)))
                    rgbs = np.array([
                        self.cnop2rgb(
                            phi=hue_angle, saturation=sat, gray_level=lum, unit="deg"
                        )[0].tolist() for (sat, lum) in cnop_vals
                    ])
                else:
                    cnop_vals = np.array(list(itertools.product(lum, sats)))
                    rgbs = np.array([
                        self.cnop2rgb(
                            phi=hue_angle, saturation=sat, gray_level=lum, unit="deg"
                        )[0].tolist() for (lum, sat) in cnop_vals
                    ])
            else:
                raise ValueError('Unknown 2d color list type "{}"'.format(axes))
        elif not isinstance(axes, str) and len(axes) == 3:
            hues = np.arange(min_hue, max_hue, hue_res).tolist()
            sat = np.arange(min_sat, max_sat, sat_res).tolist()
            lum = np.arange(min_lum, max_lum, lum_res).tolist()
            cnop_vals = np.array(list(itertools.product(hues, sat, lum)))
            rgbs = np.array([
                self.cnop2rgb(
                    phi=phi, saturation=sat, gray_level=lum, unit="deg"
                )[0].tolist() for (phi, sat, lum) in cnop_vals
            ])
        else:
            raise ValueError("Color lists can be created from 1, 2 or 3 value tuples,"
                "but currently is {} with length {}".format(axes, len(axes)))

        # resolution of N bit in [0, 1] scale
        rgb_res = 1. / np.power(2., self.bit_depth)
        rgbs = (rgbs/rgb_res).astype(np.int64)

        # generate set of unique RGB values
        uni_inds = np.sort(np.unique(rgbs, axis=0, return_index=True)[1])
        cnop_vals = cnop_vals[uni_inds]

        if "cnop_hues" in axes:
            if cnop_vals.ndim > 1:
                cnop_vals = cnop_vals.T
                cnop_vals[axes.index("cnop_hues")] %= 360.
                cnop_vals = cnop_vals.T
            else:
                cnop_vals %= 360.

        if self.bit_depth == 8 or self.bit_depth == [8, 8, 8]:
            rgbs = self.rgb2552rgb(rgbs[uni_inds, :])
        elif self.bit_depth == 10 or self.bit_depth == [10, 10, 10]:
            rgbs = self.rgb10232rgb(rgbs[uni_inds, :])
        else:
            raise ValueError("Could not convert color list with bit depth set to {}bit".format(self.bit_depth))
        rgbs = self.rgb2pp(rgbs)

        # add list to subject and colorspace
        self.color_list[axes] = dict({})
        self.color_list[axes]["res_hues"] = hue_res
        self.color_list[axes]["res_sat"] = sat_res
        self.color_list[axes]["res_lum"] = lum_res
        self.color_list[axes]["hue_angle"] = hue_angle
        self.color_list[axes]["saturation"] = saturation
        self.color_list[axes]["gray_level"] = gray_level
        self.color_list[axes]["cnop"] = cnop_vals
        self.color_list[axes]["rgb"] = rgbs
        self.op_mode = False

        return True

    def plot_iso_slant(self, path=None, directory=None, show=True):
        """
        Run input and fit a sine-function to get the iso-slant for iso-luminance plane.
        Depends on the current calibration.
        :param path: Path to file. Default is None, which results in plot_colorspace_<SUBJECT SHORT>_<DATE>.pdf file.
        :param directory: Directory, prepended to "path".
                         Default is None, which creates a directory called "colorspace_plots" in the current directory.
        :param show: If True, plot will be shown, otherwise only saved. Default is True.
        """
        try:
            from .colorspace import ColorSpace
        except ImportError:
            pass

        self.op_mode = True

        subj_short = "UNKNOWN"
        if self.subject is not None:
            subj_short = self.subject.short

        # save file options
        if not path:
            path = "colorspace_plots/plot_colorspace_{}_{}".format(subj_short, self.date)
        if directory:
            path = os.path.join(directory, os.path.basename(path))
        if "pdf" not in path:
            path += ".pdf"
        os.makedirs(os.path.split(path)[0], exist_ok=True)

        x = np.arange(0., 2.*np.pi, 0.01*np.pi)
        a = self.iso_slant["amplitude"]
        ph = self.iso_slant["phase"]

        xdata = self.iso_slant["xdata"]
        if not isinstance(xdata, np.ndarray):
            xdata = np.asarray(xdata)
        ydata = self.iso_slant["ydata"]
        if not isinstance(ydata, np.ndarray):
            ydata = np.asarray(ydata)

        f = a * np.sin(x + ph)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 4))
        ax.set_xticks(np.arange(0., 361., 45.))

        ax.plot((x/np.pi * 180.) % 360., f, c="tab:blue", label="fit")
        ax.plot((xdata/np.pi * 180.) % 360., ydata, marker='x', c="tab:red", linewidth=0, label="data")
        ax.grid()
        ax.set_xlabel("Phase [deg]")
        ax.set_ylabel("Delta Luminance [gray value [0-1]]")
        ax.legend()
        self.op_mode = False

        fig.suptitle("Iso-Luminance Fit for Subject {}".format(subj_short))
        plt.tight_layout()
        plt.savefig(path)
        if show:
            plt.show()
        plt.cla()

        return True

    def show_color_circle(self, num_col=16, gray_level=None):
        """
        Show color circle.
        Use sliders to adjust phase and amplitude of isoluminance fit.
        Then possible to save to discard, save to new file or rewrite old file.
        :param num_col: Number of colors to be shown.
        :param gray_level: Gray level.
        """

        self.op_mode = True
        if gray_level is None:
            gray_level = self.gray_level

        phis = np.linspace(0, 2 * np.pi, num_col, endpoint=False)

        # units='pix' important for slider to work
        win = visual.Window([self.monitor.currentCalib['sizePix'][0],
                             self.monitor.currentCalib['sizePix'][1]],
                             monitor=self.monitor.name, fullscr=True,
                             units='pix')

        # set background gray level
        win.colorSpace = "rgb"
        win.color = self.rgb2pp(np.array([gray_level, gray_level, gray_level]))[0]

        tb = visual.TextBox(window=win,
                            text="Set iso-slant values via sliders.\n"
                                 "ESCAPE: close without saving\n"
                                 "RETURN: save to (new) file",
                            font_size=14, font_color=[1, 1, 1],  # grid_horz_justification='center',
                            size=(0.25*win.size[0], 0.1*win.size[1]),
                            pos=(-0.33*win.size[0], -0.25*win.size[1]), units='pix')
        tb.draw()
        # win.flip()

        # add sliders
        start_amp = -self.iso_slant["amplitude"]
        amp_slider = visual.Slider(win=win, units="pix", ticks=(0.00, 0.01, 0.02, 0.03, 0.04,),
                                   labels=(0.00, 0.01, 0.02, 0.03, 0.04,),
                                   startValue=start_amp, granularity=0,
                                   pos=(0.33 * win.size[0], -0.35 * win.size[1]),
                                   size=(0.25 * win.size[0], 0.03 * win.size[1]),
                                   opacity=None, color="LightGray", fillColor="Gray", borderColor="White",
                                   colorSpace="rgb", font="Open Sans", labelHeight=12, flip=False,
                                   )
        amp_slider_title = visual.TextBox(window=win, text="Amplitude", font_size=14,
                                          font_color=[1, 1, 1], grid_horz_justification='center',
                                          pos=(0.33 * win.size[0], -0.32 * win.size[1]),
                                          size=(0.25 * win.size[0], 0.05 * win.size[1]),
                                          units='pix')

        start_phase = self.iso_slant["phase"]/np.pi * 180.
        phase_slider = visual.Slider(win=win, units="pix", ticks=(0, 45, 90, 135, 180, 225, 270, 315, 360),
                                     labels=(0, 45, 90, 135, 180, 225, 270, 315, 360), startValue=start_phase,
                                     pos=(-0.33*win.size[0], -0.35*win.size[1]),
                                     size=(0.25*win.size[0], 0.03*win.size[1]),
                                     opacity=None, color="LightGray", fillColor="Gray", borderColor="White",
                                     colorSpace="rgb", font="Open Sans", labelHeight=12, flip=False,
                                     )
        phase_slider_title = visual.TextBox(window=win, text="Phase [deg hue angle]", font_size=14,
                                            font_color=[1, 1, 1], grid_horz_justification='center',
                                            pos=(-0.33 * win.size[0], -0.32 * win.size[1]),
                                            size=(0.25 * win.size[0], 0.05 * win.size[1]),
                                            units='pix')

        # set iso_slant values accordingly
        m_rgb = self.cnop2rgb(phi=phis, gray_level=gray_level,)

        rect_size = 0.4 * win.size[0] * 2 / num_col
        radius = 0.2 * win.size[0]
        alphas = np.linspace(0, 360, num_col, endpoint=False)

        rect = visual.Rect(win=win,
                           units="pix",
                           width=int(rect_size), height=int(rect_size))
        for i_rect in range(num_col):
            rect.colorSpace = "rgb"
            rect.fillColor = m_rgb[i_rect]
            rect.lineColor = m_rgb[i_rect]
            rect.pos = misc.pol2cart(alphas[i_rect], radius)
            rect.draw()

        cr_amp = 0.
        old_rating_amp = cr_amp
        cr_phase = 0.
        old_rating_phase = cr_phase
        curr_keys = []
        while 'return' not in curr_keys and 'escape' not in curr_keys:
            if amp_slider.getRating() is not None and amp_slider.getRating() != old_rating_amp:
                cr_amp = amp_slider.getRating()
                old_rating_amp = cr_amp
                # adjust iso slant (minus because of convention in measurements)
                self.iso_slant['amplitude'] = -cr_amp
                m_rgb = self.cnop2rgb(phi=phis, gray_level=gray_level,)
            if phase_slider.getRating() is not None and phase_slider.getRating() != old_rating_phase:
                cr_phase = phase_slider.getRating()
                old_rating_phase = cr_phase
                # adjust iso slant
                self.iso_slant['phase'] = cr_phase/180. * np.pi
                m_rgb = self.cnop2rgb(phi=phis, gray_level=gray_level,)

            amp_slider.draw()
            phase_slider.draw()
            amp_slider_title.draw()
            phase_slider_title.draw()
            tb.draw()
            for i_rect in range(num_col):
                rect.colorSpace = "rgb"
                rect.fillColor = m_rgb[i_rect]
                rect.lineColor = m_rgb[i_rect]
                rect.pos = misc.pol2cart(alphas[i_rect], radius)
                rect.draw()

            win.flip()

            curr_keys = event.getKeys()

        # if return: save new values
        if 'return' in curr_keys:
            tb.text = "You want to save the current setting.\n" \
                      "Please enter the save file name/path\nand press RETURN to save it."
            tb.size = (500, 100)
            tb.pos = (0.0, 30)

            ans = visual.TextBox2(win=win, text="", color=[-1, -1, -1], font="Open Sans", size=(500, 100),
                                  pos=(0.0, -90), units='pix', editable=True)

            spath = ''
            while '\n' not in ans.text and '\r' not in ans.text:
                tb.draw()
                ans.draw()
                spath = ans.text
                win.flip()
            self.save_to_file(path=spath)
        # else (= if escape): don't save values and exit

        win.close()
        self.op_mode = False

        return True

    def show_checkerboard(self, low=1, high=13, win=None, gray_level=None,
                          saturation=None, color_list=None, update=True,
                          draw=True):
        """
        Show randomly colored checkerboard.
        :param low: Lowest number of rectangles.
        :param high: Highest number of rectangles.
        :param win: Window, which should be filled.
        :param gray_level: Gray level.
        :param saturation: saturation.
        :param color_list: List of possible colors (in rgb).
        :param update: Whether to change layout (size and colors)
        :param draw: Whether to draw it (e.g. for initialization).
        """

        self.op_mode = True
        if gray_level is None:
            gray_level = self.gray_level
        if win is None:
            win = visual.Window(fullscr=True, monitor="eDc-1")
        old_units = win.units
        win.units = "norm"
        # random number of elements
        p_num = np.random.randint(low=low, high=high, size=1)[0]

        if update:
            # create grid
            hw_ratio = win.size[0] / win.size[1]
            mask = None
            tex = None

            # INITIALIZE
            if self.checkerboard is None:
                self.checkerboard = dict({})
                for c_num in range(low, high):
                    p_w = 1. / c_num
                    p_h = p_w * hw_ratio
                    p_grid = np.mgrid[-1.:(1. + p_w):p_w, -1.:(1. + p_h):p_h].reshape(2, -1).T
                    n_dots = len(p_grid)
                    sizes = (p_w, p_h)
                    self.checkerboard[c_num] = visual.ElementArrayStim(
                        win=win,
                        nElements=n_dots,
                        units="norm",
                        xys=p_grid,
                        sizes=sizes,
                        elementTex=tex,
                        elementMask=mask
                    )
            cb = self.checkerboard[p_num]
            # get colors
            # speed process up for screensaver etc.
            if color_list is None:
                angles = np.asarray(np.random.randint(low=0, high=360, size=cb.nElements))
                cb.colors = self.rgb2pp(
                    self.cnop2rgb(phi=angles, unit="deg", saturation=saturation, gray_level=gray_level)
                )
            else:
                indices = np.asarray(np.random.randint(low=0, high=len(color_list), size=cb.nElements))
                cb.colors = list(itemgetter(*indices)(color_list))
        if draw:
            self.checkerboard[p_num].draw()
            win.flip()
        win.units = old_units
        self.op_mode = False

        return True

    def screensaver(self, gray_level=None):
        """
        Show random checkerboards as screensaver.
        :param gray_level: Gray level.
        """
        if gray_level is None:
            gray_level = self.gray_level
        win = visual.Window(fullscr=True, monitor="eDc-1")

        # Don't show cursor
        event.Mouse(visible=False)

        # create color_list
        self.create_color_list(hue_res=2.)
        color_list = self.color_list["cnop_hues"]["rgb"]

        while True:
            self.show_checkerboard(win=win, color_list=color_list)
            keys = event.waitKeys(maxWait=3)
            if keys and "escape" in keys:
                win.close()
                break

        return True

    def save_to_file(self, path=None, directory=None, filetype="yaml", absolute_paths=False, save_color_list=False):
        """
        Save colorspace data to file.
        :param path: Location of file. Default in None.
        :param directory: Directory, if file name should be filled automatically.
        :param filetype: Filetype, "json" or "yaml".
               Default is "yaml" but set to file extension if found in path.
        :param absolute_paths: If True, absolute file paths are saved. Default is False.
        :param save_color_list: If True, created color list will be saved as well. Default is False.
        :return: True.
        """

        if path is not None and "." in path:
            filetype = path.split(".")[-1]

        dt = {}
        dt.update(vars(self))
        del dt["calibration"]
        del dt["subject"]
        del dt["monitor"]
        del dt["checkerboard"]
        dt["uuid"] = str(self.uuid)
        dt["date"] = self.date.isoformat()
        if dt["calibration_path"]:
            if absolute_paths:
                dt["calibration_path"] = os.path.abspath(dt["calibration_path"])
        if dt["subject_path"]:
            if absolute_paths:
                dt["subject_path"] = os.path.abspath(dt["subject_path"])
        iso_slant = self.iso_slant
        iso_slant["xdata"] = np.asarray(iso_slant["xdata"]).tolist()
        iso_slant["ydata"] = np.asarray(iso_slant["ydata"]).tolist()
        if filetype.lower() == "json":
            iso_slant = json.dumps(iso_slant)
        dt["iso_slant"] = iso_slant
        if save_color_list:
            for key, val in self.color_list.items():
                self.color_list[key]["cnop"] = np.asarray(val["cnop"]).tolist()
                self.color_list[key]["rgb"] = np.asarray(val["rgb"]).tolist()
            if filetype.lower() == "json":
                dt["color_list"] = json.dumps(self.color_list)
        else:
            del dt["color_list"]
        dt = dict(sorted(dt.items()))

        if not path:
            path = "colorspace_{}.{}".format(self.date, filetype)
        if directory:
            path = os.path.join(directory, path)
        dump_file(dt, path, filetype)

        print("Successfully saved colorspace to file {}".format(path))

        return True

    def load_from_file(self, path=None, filetype="yaml"):
        """
        Load colorspace from file.
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
            setattr(self, a, b)

        self.uuid = uuid.UUID(str(self.uuid))
        self.date = datetime.datetime.fromisoformat(d["date"])

        if self.calibration_path:
            self.calibration = Calibration()
            self.calibration.load_from_file(self.calibration_path)
            self.monitor = Monitor(settings_path=self.calibration.monitor_settings_path)
        if self.subject_path:
            self.subject = Subject()
            self.subject.load_from_file(self.subject_path)

        if filetype.lower() == "json":
            if len(self.iso_slant) > 0:
                self.iso_slant = json.loads(self.iso_slant)
            if len(self.color_list) > 0:
                self.color_list = json.loads(self.color_list)

        print("Successfully loaded colorspace from file {}".format(path))

        return True
