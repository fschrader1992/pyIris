"""
MEASURE SPECTRA, CALIBRATE AND ADD PLOTS.
"""

import os
import datetime
from pyiris.spectrum import Spectrum
from pyiris.calibration import Calibration
from pyiris.functions import yaml2json


"""
--------------------------------
DEFINE PARAMETERS
--------------------------------

All files will be saved in YAML format here.
If you want to change to JSON or NIX, follow the instructions after the "CHANGE4JSON" tag below.
"""

# Short handle of the spectrum name.
# Directories and Filenames will be created accordingly.
# By default, it takes the current date, but you can set any name you like.
spectrum_handle = "bitdepth-10_date-" + datetime.datetime.now().strftime("%Y-%m-%d")

# Stepsize as float in range [0-1] for differences in colors.
# Each step measures spectra in single r, g, b and gray values of this intensity.
step_size = 0.1

# Minimum gray level to test.
min_gray = 0.0

# Maximum gray level to test.
max_gray = 1.0

# Number of measurement repeats for each stimulus.
nrep = 2

# Define position settings and labels. they will be added to the calibration files
# All entries within this dictionary will be run.
# Before the first run, you need to adjust the spectrometer position on screen.
positions_dict = {
    "main": {
        "xys": [[0., 0.]],
        "xy_labels": ["main"]
    },
    "left_top": {
        "xys": [[-2.475, 2.475]],
        "xy_labels": ["left_top"]
    },
    "left_bottom": {
        "xys": [[-2.475, -2.475]],
        "xy_labels": ["left_top"]
    },
    "right_top": {
        "xys": [[2.475, 2.475]],
        "xy_labels": ["left_top"]
    },
    "right_bottom": {
        "xys": [[2.475, -2.475]],
        "xy_labels": ["left_top"]
    },
    "left_horizontal": {
        "xys": [[2.475, -2.475]],
        "xy_labels": ["left_top"]
    },
    "right_horizontal": {
        "xys": [[2.475, -2.475]],
        "xy_labels": ["left_top"]
    },
}


# Dictionary defining layout options.
modes_dict = {
    "fullscreen": {
        "stim_type": None,
        "stim_size": None,
    },
    "background": {
        "stim_type": "rect",
        "stim_size": 6.,
        "background": 0.67,
    },
    "patch_black": {
        "stim_type": "circ",
        "stim_size": 2.,
        "background": 0.0,
    },
    "1DEGlum": {
        "stim_type": "circ",
        "stim_size": 1.,
        "background": 0.0,
    },
}

# Path to file with monitor settings.
# CHANGE4JSON: Change file extension to "json".
monitor_settings_path = "monitor_settings_10bit.yaml"

# Photometer port.
photo = None

# Path to CSV file with cone spectra.
cone_spectra_path = "cone_spectra.csv"

# If True, plots will be presented on screen (and saved), otherwise only saved.
show_plots = False


"""
--------------------------------
RUN MEASUREMENT AND CALIBRATION
--------------------------------
"""

# Create directories
os.makedirs("calibrations/{}/spectrum".format(spectrum_handle), exist_ok=True)
os.makedirs("calibrations/{}/calibration".format(spectrum_handle), exist_ok=True)
os.makedirs("calibrations/{}/plots".format(spectrum_handle), exist_ok=True)


"""
SPECTRUM INIT
"""
# """
# Create Spectrum
spec = Spectrum(
    photometer=photo, monitor_settings_path=monitor_settings_path
)

# Add Photometer
spec.add_pr655()
# """

for pos_label, n_pos_dict in positions_dict.items():
    adjust_at_start = True
    for mode_label, n_mode_dict in modes_dict.items():

        """
        PATHS
        """
        # Set path to new spectrum/calibration files
        spectrum_path = "calibrations/{}/spectrum/spectrum_{}_pos-{}_mode-{}".format(
            spectrum_handle, spectrum_handle, pos_label, mode_label
        )
        spectrum_plot_path = "calibrations/{}/plots/plot_spectrum_{}_pos-{}_mode-{}".format(
            spectrum_handle, spectrum_handle, pos_label, mode_label
        )

        calibration_path = "calibrations/{}/calibration/calibration_{}_pos-{}_mode-{}".format(
            spectrum_handle, spectrum_handle, pos_label, mode_label
        )
        calibration_plot_path = "calibrations/{}/plots/plot_calibration_{}_pos-{}_mode-{}".format(
            spectrum_handle, spectrum_handle, pos_label, mode_label
        )

        """
        SPECTRUM MEASUREMENT
        """
        # """
        spec.path = spectrum_path

        # Measurement Settings
        # CHANGE4JSON:
        # If you set `save_append=False`, no YAML file will be created.
        # You can either do this and uncomment one of the two lines below or use
        # the conversion after the YAML file has been saved and potentially delete the YAML file afterward.
        # Note that setting `save_append=False` will not store data after each measurement, so if something
        # goes wrong, no data will be saved.
        measure_settings = {
            "stepsize": step_size,
            "minval": min_gray,
            "maxval": max_gray,
            "n_rep": nrep,
            "adjust_at_start": adjust_at_start,
            "save_append": True,
        }

        # Update with current layout settings
        measure_settings.update(n_pos_dict)
        measure_settings.update(n_mode_dict)

        # Run Measurement
        spec.measure_colors(**measure_settings)

        if adjust_at_start:
            adjust_at_start = False

        # Uncomment if you want to save spectrum after measurement.
        # This is not necessary when save_append is True for spec.measure_colors.
        # spec.save_as_yaml()

        # CHANGE4JSON (the whole next block)
        # Uncomment the next line if you want to save the file as NIX in case you set
        # `save_append=False` above.
        # spec.save_to_file(path=f_p)

        # Alternatively, you can use the conversion function
        # spec = spec.yaml2nix(path=spectrum_path+".yaml")

        # In case you want to save the spectrum file as JSON
        # spec = yaml2json(spectrum_path+".yaml")

        # You now could also remove the yaml file
        # os.remove(spectrum_path+".yaml")

        # Plot spectra
        spec.plot_spectra(path=spectrum_plot_path, show=show_plots)
        # """

        """
        CALIBRATION
        """
        # """
        # Create Calibration
        # CHANGE4JSON: Change file extension to "json".
        cal = Calibration(mon_spectra_path=spectrum_path+'.yaml', cone_spectra_path=cone_spectra_path)

        # Perform Calibration
        # Get LMS values from spectra
        cal.calc_lms_vals()
        # Fit matrix for conversion
        cal.calibrate()

        # Save file
        # CHANGE4JSON: Change filetype to "json".
        cal.save_to_file(path=calibration_path, filetype="yaml", absolute_paths=False)

        # Plot calibration
        cal.plot(path=calibration_plot_path, show=show_plots)
        # """
