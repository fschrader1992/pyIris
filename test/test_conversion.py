import unittest
import numpy as np

from pyiris.calibration import Calibration
from pyiris.colorspace import ColorSpace


class TestConversion(unittest.TestCase):
    """
    Test color conversions.
    """

    def setUp(self):
        self.cal = Calibration(mon_spectra_path="test_spec.nix", cone_spectra_path="example/cone_spectra")
        self.cs = ColorSpace(calibration_path="cal_test.json")

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
        np.testing.assert_allclose(rgb, rgb_conv, atol=1e-7)

    def test_lms_rgb_lms(self):
        lms = np.asarray([[0., 0., 0.], [0.5, 0.5, 0.5], [1., 1., 1.]])
        rgb = self.cs.lms2rgb(lms)
        lms_conv = self.cs.rgb2lms(rgb)
        #np.testing.assert_allclose(lms, lms_conv, atol=1e-7)
