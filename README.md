# PyIris

Contents:

1. [Installation](#1-installation---building-from-source)
2. [Dependencies](#2-dependencies)
3. [Before Using the Package](#3-before-using-the-package)
4. [Commandline Tool](#4-commandline-tool)
- how to start: calibration
    - cone spectra
    - monitor spectra
    - calibration
- subjects
- colorspace
    - color conversion
    - measure an iso_slant
- create an experiment

## 1. Installation - Building From Source

You can install the package by cloning the github repository:
```shell script
git clone https://github.com/fschrader1992/pyIris
cd pyIris
python setup.py install
```

## 2. Dependencies

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
You can quit by typing `Q`, hitting `Ctrl + A` and `k` to exit screen.

## 4. Commandline Tool

|COMMAND|USE|
|---------|-----|
|`pyiris.spectrum`|Measure Spectra|
|`pyiris.calibrate`|Create rgb &lrarr; lms Calibration Matrix (Based On Spectra)|
|`pyiris.calibration.plot`|Plot Measured and Calculated Values|
|`pyiris.colorspace.isoslant`|Measure Isoslant for Calibration and Subject|
|`pyiris.colorspace.colorcircle`|Show Colorcircle for Colorspace|
|`pyiris.subject`|Add a Subject|

