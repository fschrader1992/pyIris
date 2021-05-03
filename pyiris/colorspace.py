# -*- coding: utf-8
"""
This class contains all properties/methods related to color spaces.
"""

import os
import datetime
import uuid
import json
import codecs
from pathlib import Path
from operator import itemgetter
from psychopy import event, misc, visual
from scipy import optimize

import numpy as np
import matplotlib.pylab as pl

from .monitor import Monitor
from .subject import Subject
try:
    from .calibration import Calibration
except ImportError:
    pass


def sine_fitter(x, amp, phi, off):
    """
    For iso-slant fit.
    :param x: Hue angle.
    :param amp: Amplitude.
    :param phi: Phase.
    :param off: Offset.
    :return: Sine value.
    """
    return amp * np.sin(x + phi) + off


class ColorSpace:
    """
    Class Colorspace
    """

    def __init__(self, calibration_path=None, subject_path=None, bit_depth=10, chromaticity=0.12,
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
            self.subject = self.subject.load_from_file(subject_path)
            # self.subject.colorspaces += [self]

        self.min_val = 0.00000000000001

        self.date = datetime.datetime.now()
        self.iso_slant = dict({})
        self.iso_slant["amplitude"] = 0
        self.iso_slant["phase"] = 0
        self.iso_slant["xdata"] = []
        self.iso_slant["ydata"] = []
        self.color_list = dict({})

        # dklc values
        self.chromaticity = chromaticity
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

    def dklc2lms(self, phi, gray_level=None, chromaticity=None, unit=None, s_scale=None):
        """
        Conversion of a dkl-similar value (gray/lum, phi) to a corresponding lms value.
        If a subject is given, this also depends on its iso-slant.

        :param phi: color angle(s) as list/numpy array.
        :param gray_level: luminance/gray value(s).
        :param chromaticity: Chromaticity.
        :param unit: unit for phi: rad or deg
        :param s_scale: Scaling factor for blue values.
        :return: lms values as numpy array.
        """

        phi = np.asarray(phi)
        if phi.ndim == 0:
            phi = np.asarray([phi])
        phi_len = len(phi)
        if chromaticity is None:
            chromaticity = self.chromaticity
        if phi.ndim == 0:
            chromaticity = chromaticity * np.ones(phi_len)
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
            phi = 2. * phi * np.pi/360.

        amplitude = 0.
        phase = 0.
        offset = 0.
        chrom_0 = self.chromaticity
        if self.iso_slant["amplitude"] == 0.:
            if not self.op_mode:
                print("WARNING: Amplitude of iso-slant is 0.\n"
                      "Make sure to measure subject's iso-slant with ColorSpace.measure_iso_slant.")
        else:
            amplitude = self.iso_slant["amplitude"]
            phase = self.iso_slant["phase"]
            offset = self.iso_slant["offset"]
            chrom_0 = self.iso_slant["chromaticity"]

        gray_level = np.repeat(gray_level, phi_len, axis=0)
        phase = phase * np.ones(phi_len)
        phi_lum = phi + phase

        gray_level = [gray_level + chromaticity/chrom_0 * amplitude * np.sin(phi_lum) + offset]
        gray = self.rgb2lms(np.repeat(gray_level, 3, axis = 0).T)
        gray[gray == 0] = self.min_val

        # this ratio can be adjusted
        lm_ratio = 1.0 * gray.T[0] / gray.T[1]

        vec = np.asarray([
            1.0 + chromaticity * np.cos(phi) / (1.0 + lm_ratio),
            1.0 - chromaticity * np.cos(phi) / (1.0 + 1.0/lm_ratio),
            1.0 + s_scale * chromaticity * np.sin(phi)
        ]).T

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

    def dklc2rgb(self, phi, gray_level=None, chromaticity=None, unit=None, s_scale=None):
        """
        Conversion of a dkl-similar value (gray/lum, phi) to a corresponding rgb value.
        If a subject is given, this also depends on its iso-slant.

        :param phi: color angle(s).
        :param gray_level: luminance/gray value(s).
        :param chromaticity: Chromaticity.
        :param unit: unit for phi: rad or deg
        :param s_scale: Scaling factor for blue values.
        :return: rgb values as numpy array.
        """
        lms = self.dklc2lms(phi, gray_level, chromaticity, unit, s_scale)
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

    def rgb2hex(self, rgb, cross=True):
        """
        Convert rgb values to hex.
        :param rgb: (list of) 3-tuple/numpy array with rgb values [0, 1].
        :param cross: If True, indicator "#" is added.
        :return: hex values as numpy array.
        """

        rgb255 = self.rgb2rgb255(rgb)
        hex_arr = []
        start = "#" if cross else ""
        for r in rgb255:
            hex_arr += [start + "%02x%02x%02x" % tuple(r)]
        hex_arr = np.asarray(hex_arr)
        return hex_arr

    def hex2rgb(self, hex_arr):
        """
        Convert hex values to rgb.

        :param hex_arr: (3- or 6-digit) hex values (with/-out "#") as numpy array.
        :return: (list of) 3-tuple/numpy array with rgb values [0, 1].
        """

        hs2ha = lambda t, ti, lti: int(t.lstrip("#")[ti:ti+lti] if lti == 2 else
                                       t.lstrip("#")[ti:ti+lti] + t.lstrip("#")[ti:ti+lti], 16)
        splitter = lambda t: tuple([hs2ha(t, 0, 2), hs2ha(t, 2, 2), hs2ha(t, 4, 2)])\
            if len(t) == 6 or len(t) == 7 else tuple([hs2ha(t, 0, 1), hs2ha(t, 1, 1), hs2ha(t, 2, 1)])
        rgb = self.rgb2552rgb(np.asarray(np.vectorize(splitter)(hex_arr)).T)
        return rgb

    def measure_iso_slant(self, gray_level=None, num_fit_points=10, repeats=6, lim=0.1,
                          step_size=0.001, refresh=None):
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

        win = visual.Window([self.monitor.currentCalib['sizePix'][0],
                             self.monitor.currentCalib['sizePix'][1]],
                            monitor=self.monitor.name, fullscr=True)

        # set background gray level
        win.colorSpace = "rgb"
        win.color = self.color2pp(np.array([gray_level, gray_level, gray_level]))[0]

        mouse = event.Mouse()

        info = visual.TextStim(win, pos=[-0.7, 0.95], height=0.03)
        info.autoDraw = True

        rect = visual.Rect(win, pos=[0, 0], width=0.35, height=0.5)

        for idx, phi in enumerate(randstim):
            info.text = str(idx + 1) + ' of ' + str(len(randstim)) +\
                        ' stimuli at ' + str(freq) + 'Hz'

            color = self.color2pp(self.dklc2rgb(phi, gray_level=gray_level))[0]
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
                        color = self.color2pp(self.dklc2rgb(phi, gray_level=ref_gray_level))[0]
                        if len(color[color > 1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color

                    rect.setColor(curr_color, "rgb")
                    rect.draw()

                    if event.getKeys('right'):
                        ref_gray_level = gray_level + np.ones(3) * (d_gray + step_size)
                        color = self.color2pp(self.dklc2rgb(phi, gray_level=ref_gray_level))[0]
                        if len(color[color > 1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color
                            d_gray += step_size

                    if event.getKeys('left'):
                        ref_gray_level = gray_level + np.ones(3) * (d_gray - step_size)
                        color = self.color2pp(self.dklc2rgb(phi, gray_level=ref_gray_level))[0]
                        if len(color[color < -1.]) == 0 and not np.isnan(np.sum(color)):
                            curr_color = color
                            d_gray -= step_size

                    if event.getKeys('space'):
                        break

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
        self.iso_slant["chromaticity"] = self.chromaticity
        self.iso_slant["gray_level"] = gray_level
        self.op_mode = False

    def create_color_list(self, hue_res=0.2, gray_level=None):
        """
        Generate colors that are realizable in a N-bit display and save them in a color list.
        :param hue_res: Resolution of hue angles, i.e. hue angle bins. Given in DEG!
        :param gray_level: Luminance value.
        """
        self.op_mode = True
        if gray_level is None:
            gray_level = self.gray_level

        phis = np.linspace(0, 360 - hue_res, int(360 / hue_res))
        conv_phi = [self.dklc2rgb(phi=phi, gray_level=gray_level, unit="deg") for phi in phis]
        rgb = [c_phi for c_phi in conv_phi]
        # resolution of N bit in [0, 1] scale
        rgb_res = 1. / np.power(2., self.bit_depth)
        sel_rgb = []
        sel_phi = []
        rgb_len = len(rgb)
        it = iter(list(range(0, len(rgb) + 2)))

        for idx in it:
            if idx < rgb_len:
                sel_rgb.append(rgb[idx][0])
                sel_phi.append(phis[idx])
                # prevent multiple entries
                if abs(rgb[idx][0][1] - rgb[(idx + 1) % rgb_len][0][1]) <= rgb_res:
                    next(it)
                if abs(rgb[idx][0][1] - rgb[(idx + 2) % rgb_len][0][1]) <= rgb_res:
                    next(it)
                    next(it)

        # add list to subject and colorspace
        self.color_list[hue_res] = dict({})
        self.color_list[hue_res]["hue_angles"] = sel_phi
        self.color_list[hue_res]["rgb"] = sel_rgb
        self.op_mode = False

    def plot_iso_slant(self):
        """
        Run input and fit a sine-function to get the iso-slant for iso-luminance plane.
        Depends on the current calibration.
        """

        self.op_mode = True
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
        self.op_mode = False

    def show_color_circle(self, num_col=16, gray_level=None):
        """
        Show color circle.
        :param num_col: Number of colors to be shown.
        :param gray_level: Gray level.
        """

        self.op_mode = True
        if gray_level is None:
            gray_level = self.gray_level

        phis = np.linspace(0, 2 * np.pi, num_col, endpoint=False)
        m_rgb = self.dklc2rgb(phi=phis, gray_level=gray_level,)

        win = visual.Window(size=[800, 600], monitor="eDc-1", fullscr=False)
        # set background gray level
        win.colorSpace = "rgb"
        win.color = self.color2pp(np.array([gray_level, gray_level, gray_level]))[0]
        win.flip()

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
        self.op_mode = False

    def show_checkerboard(self, low=1, high=13, win=None, gray_level=None,
                          chromaticity=None, color_list=None, update=True,
                          draw=True):
        """
        Show randomly colored checkerboard.
        :param low: Lowest number of rectangles.
        :param high: Highest number of rectangles.
        :param win: Window, which should be filled.
        :param gray_level: Gray level.
        :param chromaticity: Chromaticity.
        :param color_list: List of possible colors (in rgb).
        :param update: Whether or not to change layout (size and colors)
        :param draw: Whether or not to draw it (e.g. for initialization).
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
            else:
                cb = self.checkerboard[p_num]
                # get colors
                # speed process up for screensaver etc.
                if color_list is None:
                    angles = np.asarray(np.random.randint(low=0, high=360, size=cb.nElements))
                    cb.setColors(self.color2pp(self.dklc2rgb(phi=angles, unit="deg",
                                                         chromaticity=chromaticity,
                                                         gray_level=gray_level)))
                else:
                    indices = np.asarray(np.random.randint(low=0, high=len(color_list), size=cb.nElements))
                    cb.setColors(list(itemgetter(*indices)(color_list)))
        if draw:
            self.checkerboard[p_num].draw()
            win.flip()
        win.units = old_units
        self.op_mode = False

    def screensaver(self, gray_level=None):
        """
        Show random checkerboards as screensaver.
        :param gray_level: Gray level.
        """
        if gray_level is None:
            gray_level = self.gray_level
        win = visual.Window(fullscr=True, monitor="eDc-1")

        # create color_list
        if 2. not in self.color_list.keys():
            self.create_color_list(hue_res=2., gray_level=gray_level)
        color_list = self.color_list[2.]["rgb"]

        while True:
            self.show_checkerboard(win=win, color_list=color_list)
            keys = event.waitKeys(maxWait=3)
            if keys and "escape" in keys:
                win.close()
                break

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
        del dt["monitor"]
        dt["uuid"] = str(self.uuid)
        dt["date"] = str(self.date)
        if dt["calibration_path"]:
            dt["calibration_path"] = str(Path(dt["calibration_path"]).resolve())
        if dt["subject_path"]:
            dt["subject_path"] = str(Path(dt["subject_path"]).resolve())
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
            self.monitor = Monitor(settings_path=self.calibration.monitor_settings_path)
        if self.subject_path:
            self.subject = Subject()
            self.subject.load_from_file(self.subject_path)

        self.iso_slant = json.loads(self.iso_slant)
        self.color_list = json.loads(self.color_list)

        print("Successfully loaded colorspace from file {}".format(path))
