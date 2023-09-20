# PyIris

PyIris is a tool that helps creating visual experiments with psychopy.
It is possible to calibrate a monitor in order to be able to set LSM-colors 
that can be adjusted to individual subjects.
Tested for python versions > 3.6.


## Contents

1. [Installation](#1-installation---building-from-source)
2. [Dependencies](#2-dependencies)
3. [Before Using the Package](#3-before-using-the-package)
4. [Commandline Tool](#4-commandline-tool)
5. [File Structure](#5-file-structure)

## 1. Installation - Building From Source

You can install the package by cloning the github repository:
```shell script
git clone https://github.com/fschrader1992/pyIris
cd pyIris
python setup.py install
```

## 2. Dependencies

- psychopy
- numpy
- scipy
- matplotlib
- pandas
- symfit
- h5py
- pyyaml
- nixio >= 1.5.0b1 (otherwise writing to odML-like metadata will not work correctly)

## 3. Before Using the Package

In order to use the PR655 photometer, we need to set set the device driver to `usbserial`. You can check which driver 
is used by typing `lsusb -t` (or whichever you prefer) into the terminal. As the standard is `cdc-acm`, we need to 
blacklist it. This can be done by creating the file `etc/modprobe.d/pr655.conf` with the lines (cf. modprobe.conf(5)):
```shell script
#PR655 should use usbserial
blacklist cdc-acm
```
After adding this, a reboot is necessary. Check again with `lsusb -t`. If the driver is set right. If no driver is
shown, you need to add `usbserial` via the terminal:
```shell script
modprobe usbserial vendorID=XXXX productID=XXXX
```
You can get the two variables by searching for it with `usb-devices` in the terminal. If this does not work, you can
also try the following: Unplug the device and execute the following two commands:
```shell script
sudo modprobe usbserial
sudo sh -c "echo [vendorID] [productID] >/sys/bus/usb-serial/drivers/generic/new_id"
```
If you plug your photometer back in and check with `lsusb -t` you now should see the right driver.
If this works, the photometer is now connected to port `/dev/ttyUSB0`. If you don't want to run the commands that need 
to access the usb port with sudo, you can add the rights to your current user via the terminal:
```shell script
sudo usermod -a -G dialout USER
```
Alternatively you can set the access to the port by 
```shell script
sudo chmod /dev/ttyUSB0 0666
```
You can see, whether you are able to connect to the device by using screen:
```shell script
screen /dev/ttyUSB0 9600
```
(where 9600 is the baudrate, change, if needed) and then type `PHOTOMETER`. The photometer should now be in remote mode.
You can quit by pressing `Q`, then `Ctrl + A` and `k` to exit screen.

## 4. Commandline Tool

|COMMAND|USE|
|---------|-----|
|`pyiris.screensaver`|Show Screensaver (Exit by Pressing ESCAPE)|
|`pyiris.spectrum`|Measure Spectra|
|`pyiris.calibrate`|Create rgb &lrarr; lms Calibration Matrix (Based On Spectra)|
|`pyiris.calibration.plot`|Plot Measured and Calculated Values|
|`pyiris.subject`|Add a Subject|
|`pyiris.colorspace.isoslant`|Measure Isoslant for Calibration and Subject|
|`pyiris.colorspace.colorlist`|Add List of Colors on Isoluminance Plane|
|`pyiris.colorspace.colorcircle`|Show Colorcircle for Colorspace|

The commands in detail:

```
pyiris.screensaver [-h/--help] [-g/--graylevel]
    -h/--help               Show help.
    -g/--graylevel          Gray level [0-1]. Default is 0.66.
```

```
pyiris.spectrum [-h/--help] -s/--stepsize [-M/--monitor] -P/--photometer [-d/--directory]
                [-p/--path]
    -h/--help               Show help.
    -M/--monitor            
    -P/--photometer         
    -d/--directory          Directory the spectrum file should be stored in. If no filename is 
                            specified in the path variable, it is saved as spectrum_DATETIME.nix.
    -p/--path               Path the file should be stored in. This can be either the name of the
                            spectrum file (works in combination with directory as well) or 
                            the full path. If it is not given, the spectrum will be saved in the
                            current directory under spectrum_DATETIME.nix.
```

```
pyiris.calibrate [-h/--help] -S/--spectra -C/--cones [-d/--directory] [-p/--path] 
    -h/--help               Show help.
    -S/--spectra            Path to file with measured (monitor) spectra.
    -C/--cones              Path to file with cone spectra (cf. example in test/resources).
    -d/--directory          Directory the calibration file should be stored in. If no filename is 
                            specified in the path variable, it is saved as 
                            calibration_DATETIME.json.
    -p/--path               Path the file should be stored in. This can be either the name of the
                            calibration file (works in combination with directory as well) or 
                            the full path. If it is not given, the calibration will be saved in
                            the current directory under calibration_DATETIME.json.
```

```	
pyiris.plot_calibration [-h/--help] -p/--path
    -h/--help               Show help.
    -p/--path               Path to file with calibration to plot.
```

```
pyiris.subject [-h/--help] [-s/--short] [-N/--name] [-S/--surname] [-b/--birthday]
               [-n/--notes] [-d/--directory] [-p/--path] 
    -h/--help	            Show help.
    -s/--short              Subject short, if given it is used in the filename.
    -N/--name               Name of subject.
    -S/--surname            Surname of subject.
    -b/--birthday           Birthdate of subject (as string).
    -n/--notes              Further notes you want to add.
    -d/--directory          Directory the subject file should be stored in. If no filename is 
                            specified in the path variable, it is saved as subject_UUID/SHORT.json.
    -p/--path               Path the file should be stored in. This can be either the name of the
                            subject file (works in combination with directory as well) or 
                            the full path. If it is not given, the subject will be saved in the
                            current directory under subject_UUID/SHORT.json.
```

```	
pyiris.colorspace.isoslant [-h/--help] [-C/--calibration] [-S/--subject] [-b/--bitdepth]
                           [-c/--chromaticity] [-g/--graylevel] [-u/--unit] [-s/--sscale]
                           [-d/--directory] [-p/--path] 
    -h/--help               Show help.
    -C/--calibration        Path to file with calibration. Without this you can only use functions
                            that convert between different rgb-colorspaces.
    -S/--subject            Path to file with subject.
    -b/--bitdepth           Bit depth of one color of the current monitor (e.g. 8 or 10).
                            Default is 10bit.
    -c/--chromaticity       Chromaticity (contrast) for colorcircle, maximum 0.36. Default is 0.12.
    -g/--graylevel          Gray level [0-1]. Default is 0.66.
    -s/--sscale             Scaling for S-cone values for better viewing. Default is 2.6.
    -d/--directory          Directory the calibration file should be stored in. If no filename is 
                            specified in the path variable, it is saved as 
                            calibration_DATETIME.json.
    -p/--path               Path the file should be stored in. This can be either the name of the
                            calibration file (works in combination with directory as well) or 
                            the full path. If it is not given, the calibration will be saved in
                            the current directory under calibration_DATETIME.json.
```

```
pyiris.colorspace.colorlist [-h/--help] -p/--path [-r/--resolution] [-g/--graylevel]
    -h/--help               Show help.
    -p/--path               Path to file with colorspace to plot circle for.
    -r/--resolution         Resoultion of hue angle in DEGREE. Default is 0.2.
    -g/--graylevel          Gray level [0-1]. Default is 0.66.
```

```
pyiris.colorspace.colorcircle [-h/--help] [-n/--num] -p/--path
    -h/--help               Show help.
    -n/--num                Number of patches/hue angles to be shown. Default is 16.
    -p/--path               Path to file with colorspace to plot circle for.
```

## 5. File Structure

You can name and store files as you like. You can also use the default names of files,
containing the file creation date or, in case of subjects, its short name.

In order to use the tool, you have to give two files. One provides the spectra of cones.
It is a csv file containing (at least) the following columns, which will be found by their names:
`wavelength`, `L`, `M` and `S`.
If you don't want to provide your own file, you can use the file found at 
`test/resources/cone_spectra.csv`.

The other file you need is a yaml file containing the monitor settings. 
You can use the file at `test/resources/monitor_settings.yaml` as template.
