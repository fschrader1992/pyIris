import unittest
import os
import numpy as np

from pyiris.calibration import Calibration
from pyiris.colorspace import ColorSpace


class TestConversion(unittest.TestCase):
    """
    Test color conversions.
    """

    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cal_path = os.path.join(dir_path, "resources/calibration_test.json")
        cs_path = os.path.join(dir_path, "resources/colorspace_test.json")
        self.cal = Calibration()
        self.cal.load_from_file(path=cal_path)
        self.cs = ColorSpace()
        self.cal.load_from_file(path=cs_path)
        self.cs.calibration = self.cal

    def test_color2pp(self):
        colors = np.asarray([[0., 0., 0.], [0.5, 0.5, 0.5], [1., 1., 1.]])
        pp_colors = np.asarray([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]])
        pp_colors_conv = self.cs.color2pp(colors)
        np.testing.assert_allclose(pp_colors, pp_colors_conv, atol=1e-8)

    def test_pp2color(self):
        pp_colors = np.asarray([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]])
        colors = np.asarray([[0., 0., 0.], [0.5, 0.5, 0.5], [1., 1., 1.]])
        colors_conv = self.cs.pp2color(pp_colors)
        np.testing.assert_allclose(colors, colors_conv, atol=1e-8)

    def test_rgb_lms_rgb(self):
        rgb = np.asarray([[0., 0., 0.], [0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5], [1., 1., 1.]])
        lms = self.cs.rgb2lms(rgb)
        rgb_conv = self.cs.lms2rgb(lms)
        np.testing.assert_allclose(rgb, rgb_conv, atol=1e-6)

    def test_lms_rgb_lms(self):
        lms = np.asarray([[0., 0., 0.], [0.5, 0.5, 0.5], [1., 1., 1.]])
        rgb = self.cs.lms2rgb(lms)
        lms_conv = self.cs.rgb2lms(rgb)
        np.testing.assert_allclose(lms, lms_conv, atol=1e-8)

    def test_lms_dklc(self):
        lms = np.asarray([[0., 0., 0.], [0.5, 0.5, 0.5], [1., 1., 1.]])
        lms = np.asarray([0.5, 0.5, 0.5])
        #dklc = self.cs.lms2dklc(lms)
        #lms_conv = self.cs.dklc2lms(dklc)
        #np.testing.assert_allclose(lms, lms_conv, atol=1e-8)

    def test_dklc_lms(self):
        thetas = np.asarray([0., 0.5*np.pi, 1.0*np.pi, 1.5*np.pi])
        lms_s = self.cs.dklc2lms(thetas)
        theta = [0.5*np.pi]
        lms = self.cs.dklc2lms(theta)

    def test_rgb_rgb255(self):
        rgb = np.asarray([[0., 0., 0.], [0.5, 0.5, 0.5], [1., 1., 1.]])
        rgb255 = np.asarray([[0, 0, 0], [128, 128, 128], [255, 255, 255]])
        rgb255_conv = self.cs.rgb2rgb255(rgb)
        np.testing.assert_allclose(rgb255, rgb255_conv, atol=1e-8)

    def test_rgb255_rgb(self):
        rgb255 = np.asarray([[0, 0, 0], [128, 128, 128], [255, 255, 255]])
        rgb = np.asarray([[0., 0., 0.], [0.501961, 0.501961, 0.501961], [1., 1., 1.]])
        rgb_conv = self.cs.rgb2552rgb(rgb255)
        np.testing.assert_allclose(rgb, rgb_conv, atol=1e-6)
    
    def test_rgb_rgb1023(self):
        rgb = np.asarray([[0., 0., 0.], [0.5, 0.5, 0.5], [1., 1., 1.]])
        rgb1023 = np.asarray([[0, 0, 0], [512, 512, 512], [1023, 1023, 1023]])
        rgb1023_conv = self.cs.rgb2rgb1023(rgb)
        np.testing.assert_allclose(rgb1023, rgb1023_conv, atol=1e-8)

    def test_rgb1023_rgb(self):
        rgb1023 = np.asarray([[0, 0, 0], [512, 512, 512], [1023, 1023, 1023]])
        rgb = np.asarray([[0., 0., 0.], [0.500489, 0.500489, 0.500489], [1., 1., 1.]])
        rgb_conv = self.cs.rgb10232rgb(rgb1023)
        np.testing.assert_allclose(rgb, rgb_conv, atol=1e-6)

    def test_rgb_lms_rgb_tuple(self):
        rgb = (0.5, 0.5, 0.5)
        lms = self.cs.rgb2lms(rgb)
        rgb_conv = self.cs.lms2rgb(lms)
        np.testing.assert_allclose(np.asarray([rgb]), rgb_conv, atol=1e-8)

    def test_lms_rgb_lms_tuple(self):
        lms = (0.5, 0.5, 0.5)
        rgb = self.cs.lms2rgb(lms)
        lms_conv = self.cs.rgb2lms(rgb)
        np.testing.assert_allclose(np.asarray([lms]), lms_conv, atol=1e-8)

    def test_color2pp_tuple(self):
        colors = (0.0, 0.5, 1.0)
        pp_colors = np.asarray([[-1.0, 0.0, 1.0]])
        pp_colors_conv = self.cs.color2pp(colors)
        np.testing.assert_allclose(pp_colors, pp_colors_conv, atol=1e-8)

    def test_pp2color_tuple(self):
        pp_colors = (-1.0, 0.0, 1.0)
        colors = np.asarray([[0.0, 0.5, 1.0]])
        colors_conv = self.cs.pp2color(pp_colors)
        np.testing.assert_allclose(colors, colors_conv, atol=1e-8)

    def test_rgb_rgb255_tuple(self):
        rgb = (1., 1., 1.)
        rgb255 = np.asarray([[255, 255, 255]])
        rgb255_conv = self.cs.rgb2rgb255(rgb)
        np.testing.assert_allclose(rgb255, rgb255_conv, atol=1e-8)

    def test_rgb255_rgb_tuple(self):
        rgb255 = (255, 255, 255)
        rgb = np.asarray([[1., 1., 1.]])
        rgb_conv = self.cs.rgb2552rgb(rgb255)
        np.testing.assert_allclose(rgb, rgb_conv, atol=1e-6)

    def test_rgb_rgb1023_tuple(self):
        rgb = (1., 1., 1.)
        rgb1023 = np.asarray([[1023, 1023, 1023]])
        rgb1023_conv = self.cs.rgb2rgb1023(rgb)
        np.testing.assert_allclose(rgb1023, rgb1023_conv, atol=1e-8)

    def test_rgb1023_rgb_tuple(self):
        rgb1023 = (1023, 1023, 1023)
        rgb = np.asarray([[1., 1., 1.]])
        rgb_conv = self.cs.rgb10232rgb(rgb1023)
        np.testing.assert_allclose(rgb, rgb_conv, atol=1e-6)

    def test_rgb_lms_rgb_list(self):
        rgb = [0.5, 0.5, 0.5]
        lms = self.cs.rgb2lms(rgb)
        rgb_conv = self.cs.lms2rgb(lms)
        np.testing.assert_allclose(np.asarray([rgb]), rgb_conv, atol=1e-8)

    def test_lms_rgb_lms_list(self):
        lms = [0.5, 0.5, 0.5]
        rgb = self.cs.lms2rgb(lms)
        lms_conv = self.cs.rgb2lms(rgb)
        np.testing.assert_allclose(np.asarray([lms]), lms_conv, atol=1e-8)

    def test_color2pp_list(self):
        colors = [0.0, 0.5, 1.0]
        pp_colors = np.asarray([[-1.0, 0.0, 1.0]])
        pp_colors_conv = self.cs.color2pp(colors)
        np.testing.assert_allclose(pp_colors, pp_colors_conv, atol=1e-8)

    def test_pp2color_list(self):
        pp_colors = [-1.0, 0.0, 1.0]
        colors = np.asarray([[0.0, 0.5, 1.0]])
        colors_conv = self.cs.pp2color(pp_colors)
        np.testing.assert_allclose(colors, colors_conv, atol=1e-8)

    def test_rgb_rgb255_list(self):
        rgb = [1., 1., 1.]
        rgb255 = np.asarray([[255, 255, 255]])
        rgb255_conv = self.cs.rgb2rgb255(rgb)
        np.testing.assert_allclose(rgb255, rgb255_conv, atol=1e-8)

    def test_rgb255_rgb_list(self):
        rgb255 = [255, 255, 255]
        rgb = np.asarray([[1., 1., 1.]])
        rgb_conv = self.cs.rgb2552rgb(rgb255)
        np.testing.assert_allclose(rgb, rgb_conv, atol=1e-6)

    def test_rgb_rgb1023_list(self):
        rgb = [1., 1., 1.]
        rgb1023 = np.asarray([[1023, 1023, 1023]])
        rgb1023_conv = self.cs.rgb2rgb1023(rgb)
        np.testing.assert_allclose(rgb1023, rgb1023_conv, atol=1e-8)

    def test_rgb1023_rgb_list(self):
        rgb1023 = [1023, 1023, 1023]
        rgb = np.asarray([[1., 1., 1.]])
        rgb_conv = self.cs.rgb10232rgb(rgb1023)
        np.testing.assert_allclose(rgb, rgb_conv, atol=1e-6)

    def test_rgb255_hex(self):
        rgb255 = np.asarray([[0., 0., 0.], [128., 128., 128.],
                             [255., 0., 0.], [0., 255., 0.], [0., 0., 255.],
                             [255., 255., 255.]])
        hex_arr = np.asarray(["#000000", "#808080", "#ff0000",
                              "#00ff00", "#0000ff", "#ffffff"])
        rgb255_conv = self.cs.rgb2552hex(rgb255)
        np.testing.assert_equal(hex_arr, rgb255_conv)

    def test_rgb_hex(self):
        rgb = np.asarray([[0., 0., 0.], [0.5, 0.5, 0.5],
                          [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                          [1., 1., 1.]])
        hex_arr = np.asarray(["#000000", "#808080", "#ff0000",
                              "#00ff00", "#0000ff", "#ffffff"])
        rgb_conv = self.cs.rgb2hex(rgb)
        np.testing.assert_equal(hex_arr, rgb_conv)

    def test_hex_rgb255(self):
        hex_arr = np.asarray(["#00ff00", "#0f0",
                              "00ff00", "0f0",
                              "#000000", "#808080", "#ffffff"])
        rgb = np.asarray([[0., 255., 0.], [0., 255., 0.], [0., 255., 0.], [0., 255., 0.],
                          [0., 0., 0.], [128., 128., 128.], [255., 255., 255.]])
        hex_conv = self.cs.hex2rgb255(hex_arr)
        np.testing.assert_allclose(rgb, hex_conv, atol=1e-6)

    def test_hex_rgb(self):
        hex_arr = np.asarray(["#00ff00", "#0f0",
                              "00ff00", "0f0",
                              "#000000", "#808080", "#ffffff"])
        rgb = np.asarray([[0., 1., 0.], [0., 1., 0.],
                          [0., 1., 0.], [0., 1., 0.],
                          [0., 0., 0.], [0.50196078, 0.50196078, 0.50196078], [1., 1., 1.]])
        hex_conv = self.cs.hex2rgb(hex_arr)
        print("CONV", hex_conv)
        print("NNV", rgb)
        np.testing.assert_allclose(rgb, hex_conv, atol=1e-6)
