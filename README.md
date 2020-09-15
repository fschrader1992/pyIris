# PyIris

Contents:

1. [Installation](#1-installation---building-from-source)
2. [Before Using the Package](#2-before-using-the-package)
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

    git clone https://github.com/fschrader1992/pyIris
    cd pyIris
    python setup.py install

## 2. Before Using the Package

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
You can get the two variables from `lsusb` as well. Now the photometer should be connected to port `/dev/ttyUSB0`. 
If you don't want to run the commands that need to access the usb port with sudo, you can add the rights to your current
user via the terminal:
```shell script
sudo usermod -a -G dialout USER
```
Alternatively you can set the access tot he port by 
```shell script
sudo chmod /dev/ttyUSB0 0666
```
You can see, whether you are able to connect to the device by using screen:
```shell script
screen /dev/ttyUSB0 9600
```
(where 9600 is the baudrate, change, if needed) and then type `PHOTOMETER`. The photometer should now be in remote mode.
You can quit by typing `Q` and hitting `Ctrl + A` and `k` to exit screen.

## Commandline Tool

## Dependencies
