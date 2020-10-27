
import os
import argparse

from .calibration import Calibration
from .colorspace import ColorSpace
from .spectrum import Spectrum
from .subject import Subject


def spectrum():
    parser = argparse.ArgumentParser(description="Measure spectra.")
    parser.add_argument("-s", "--stepsize", type=float, metavar="",
                        help="Stepsize as float in range [0-1] for differences in colors. "
                             "Each step measures spectra in single r, g, b and gray "
                             "values of this intensity.")
    parser.add_argument("-M", "--monitor", metavar="",
                        help="Path to file with monitor settings.")
    parser.add_argument("-P", "--photometer", metavar="",
                        help="Port of photometer.")
    parser.add_argument("-d", "--directory", metavar="",
                        help="Directory for spectrum file.")
    parser.add_argument("-p", "--path", metavar="", help="Path for spectrum file.")

    args = parser.parse_args()

    photo = args.photometer if args.photometer else None
    spectrum = Spectrum(photometer=photo, stepsize=args.stepsize,
                        monitor_settings_path=args.monitor)

    spectrum.add_pr655()
    spectrum.measure_colors()

    f_d = args.directory if args.directory else None
    f_p = args.path if args.path else None
    spectrum.save_to_file(directory=f_d, path=f_p)


def calibrate():
    parser = argparse.ArgumentParser(description="Determine calibration matrix.")
    parser.add_argument("-S", "--spectra", metavar="",
                        help="Path to file with measured spectra object.")
    parser.add_argument("-C", "--cones", metavar="",
                        help="Path to file with cone spectra.")
    parser.add_argument("-d", "--directory", metavar="",
                        help="Directory for calibration file.")
    parser.add_argument("-p", "--path", metavar="",
                        help="Path for calibration file.")

    args = parser.parse_args()

    photo = args.photometer if args.photometer else None
    calibration = Calibration(mon_spectra_path=args.spectra, cone_spectra_path=args.cones)

    calibration.calc_lms_vals()
    calibration.calibrate()

    f_d = args.directory if args.directory else None
    f_p = args.path if args.path else None
    calibration.save_to_file(directory=f_d, path=f_p)


def plot_calibration():
    parser = argparse.ArgumentParser(description="Plot calibration.")
    parser.add_argument("-p", "--path", metavar="", help="Path for calibration file.")

    args = parser.parse_args()
    calibration = Calibration()
    calibration.load_from_file(args.path)
    calibration.plot()


def measure_iso_slant():
    parser = argparse.ArgumentParser(description="Measure isoslant.")
    parser.add_argument("-C", "--calibration", metavar="",
                        help="Path to file with measured spectra object.")
    parser.add_argument("-S", "--subject", metavar="",
                        help="Path to file with subject data.")
    parser.add_argument("-b", "--bitdepth", metavar="", type=int, default=10,
                        help="Color bit-depth.")
    parser.add_argument("-c", "--chromaticity", metavar="", type=float, default=0.12,
                        help="Chromaticity (<= 0.36).")
    parser.add_argument("-g", "--graylevel", metavar="", type=float, default=0.66,
                        help="Gray level.")
    parser.add_argument("-u", "--unit", metavar="", help="Unit for hue angle (rad or deg).")
    parser.add_argument("-s", "--sscale", metavar="", dtype=float, default=2.6,
                        help="Scale S-cone values for better viewing.")

    parser.add_argument("-d", "--directory", metavar="", help="Directory for colorspace file.")
    parser.add_argument("-p", "--path", metavar="", help="Path for colorspace file.")

    args = parser.parse_args()

    photo = args.photometer if args.photometer else None
    color_space = ColorSpace(calibration_path=args.calibration, subject_path=args.subject,
                             bit_depth=args.bitdepth, chromaticity=args.chromaticity,
                             gray_lavel=args.graylevel, unit=args.unit, s_scale=args.sscale)

    f_d = args.directory if args.directory else None
    f_p = args.path if args.path else None
    color_space.save_to_file(directory=f_d, path=f_p)


def color_circle():
    parser = argparse.ArgumentParser(description="Show color circle.")
    parser.add_argument("-p", "--path", metavar="", help="Path for colorspace file.")
    parser.add_argument("-n", "--num", metavar="", type=int, default=16,
                        help="Number of hue angles to show.")
    args = parser.parse_args()

    color_space = ColorSpace()
    color_space.load_from_file(args.path)
    color_space.show_color_circle(num_col=args.num)


def color_list():
    parser = argparse.ArgumentParser(description="Create color list for resolution.")
    parser.add_argument("-p", "--path", metavar="", help="Path for colorspace file.")
    parser.add_argument("-r", "--resolution", metavar="", default=0.2,
                        help="Resolution for hue angle in degree.")
    parser.add_argument("-g", "--graylevel", metavar="", type=float, default=0.66,
                        help="Gray level.")
    args = parser.parse_args()

    color_space = ColorSpace()
    color_space.load_from_file(args.path)
    color_space.create_color_list(hue_res=args.resolution, gray_level=args.graylevel)


def screensaver():
    """
    Show the screensaver (exit by pressing "escape").
    """
    parser = argparse.ArgumentParser(description="Show the screensaver.")
    parser.add_argument("-g", "--graylevel", metavar="", type=float, default=0.5,
                        help="Gray level.")
    args = parser.parse_args()

    color_space = ColorSpace()
    color_space.calibration = Calibration()
    color_space.calibration.set_mock_values()
    color_space.screensaver(gray_level=args.graylevel)


def subject():
    parser = argparse.ArgumentParser(description="Add a Subject.")
    parser.add_argument("-s", "--short", metavar="", required=True,
                        help="Short name that refers to subject. "
                             "Used for filename.")
    parser.add_argument("-N", "--name", metavar="", help="Name of subject.")
    parser.add_argument("-S", "--surname", metavar="", help="Surname of subject.")
    parser.add_argument("-b", "--birthday", metavar="", help="Birthday of subject.")
    parser.add_argument("-n", "--notes", metavar="", help="Notes for subject.")
    parser.add_argument("-d", "--directory", metavar="", help="Directory for subject file.")
    parser.add_argument("-p", "--path", metavar="", help="Path for subject file.")

    args = parser.parse_args()

    subject = Subject(short=args.short, name=args.name, surname=args.surname,
                birthday=args.birthday, notes=args.notes)
    f_d = args.directory if args.directory else None
    f_p = args.path if args.path else None
    subject.save_to_file(directory=f_d, path=f_p)
