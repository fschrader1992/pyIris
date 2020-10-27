import sys
import numpy as np

import time

import matplotlib.pylab as pl

from psychopy import visual, core, hardware, event
from psychopy.hardware.pr import PR655
#'''
from pyiris.spectrum import Spectrum
from pyiris.pr655 import PR655
from pyiris.calibration import Calibration
from pyiris.colorspace import ColorSpace
from pyiris.subject import Subject
'''
s = Spectrum(stepsize=1.)
#s.add_pr655()
#s.colors = [[1., 1., 1.]]
#s.measure_colors()
s.save_to_file("test_loc.nix")
time.sleep(2)
'''
#s2 = Spectrum()
#s2.load_from_file("spectrum_mycomp_2.nix")

#s2.save_to_file("spectrum_mycomp_2.nix")
'''
sq = ["[0.65 0. 0.]", "[0. 0.65 0.]", "[0. 0. 0.65]", "[0.65 0.65 0.65]", "[0.6 0.6 0.6]"]
s2.photometer = None
print(s2.names)
l = []
for n in s2.names:
    if n not in sq:
        l += [n]
s2.names = l
'''

#pl.plot(s2.spectra["[0. 0. 1.]", "power"])
#pl.plot(s2.spectra["[1. 1. 1.]", "power"])
#pl.show()
#'''
'''
s2 = Spectrum()
s2.load_from_file("test_spec.nix")
s2.add_monitor_settings("./test/resources/monitor_settings.yaml")
print(s2.monitor["vendor"])
#s2.save_to_file("test_spec.nix")
#'''
'''
p = "test/resources/spectrum_test.nix"
#p = "spectrum_mycomp.nix"
#c1 = Calibration(mon_spectra_path=p, cone_spectra_path="example/cone_spectra_old")
c1 = Calibration(mon_spectra_path=p, cone_spectra_path="test/resources/cone_spectra.csv")
# -> if given load here already
# -> make optional params
c1.calc_lms_vals()
c1.calibrate()
# lms = c1.rgb2lms_gamma([0.5, 0.5, 0.5])
# print(lms)
c1.plot()
c1.save_to_file("calibration_test.json")
#'''

#'''
c2 = Calibration()
c2.uuid = str(c2.uuid)
c2.load_from_file(path="test/resources/calibration_test.json")
print(c2.date, c2.calibration_matrix)
c2.plot()

# Test: load and assert that values are equal

#'''
#'''
#cs1 = ColorSpace(calibration_path="test/resources/calibration_test.json")
#cs1.measure_iso_slant(num_fit_points=4, repeats=2, step_size=0.1, gray_level=0.5)
#cs1.plot_iso_slant()
#cs1.show_colorcircle()

#cs1.create_color_list(hue_res=360./16.)
#print(cs1.color_list)
#cs1.show_color_circle()
#cs1.screensaver()

#print(cs1.color_list)
#cs1.save_to_file(path="colorspace_test.json")
#'''
'''
cs2 = ColorSpace()
cs2.load_from_file("colorspace_test.json")

win = visual.Window(size=[1200, 200], colorSpace="rgb")

rect_size = 30.
i = 0
#print(cs2.color_list.keys())
for rgb in cs2.color_list['22.5']["rgb"]:
    f_c = 2.*np.asarray(rgb)-1.
    rect = visual.Rect(win=win,
                       units="pix",
                       width=int(rect_size), height=int(rect_size),
                       fillColor=f_c, lineColor=f_c)

    #rect.fillColorSpace = "rgb"
    #rect.fillColor = f_c
    rect.pos = [i*2*rect_size - 500, 0]
    rect.draw()
    i += 1

win.flip()
event.waitKeys()
win.close()
#cs2.plot_iso_slant()

#print(cs2.lms_center)
#cs2.show_color_circle()
#'''
'''
#cs1.calibration.plot()
#print("CAL")
#print(cs1.calibration.calibration_matrix)
#print(cs1.calibration.inv_calibration_matrix)
print("LMS")
#print(np.asarray([[0.05, 0., 0.], [0., 0.05, 0.05]]))
lmss = cs1.rgb2lms(rgb=np.asarray([[0.5, 0., 0.], [0., 0.5, 0.5]]))
print(lmss)
print(cs1.lms2rgb(lms=lmss))

print("RGB")
lmss = cs1.lms2rgb(lms=np.asarray([[0.5, 0.4, 0.5], [0.2, 0.2, 0.1]]))
print(lmss)
print(cs1.rgb2lms(rgb=lmss))
'''
'''
win_h = 800
win_w = 600
win = visual.Window([win_h, win_w], monitor="eDc-1")
rect = visual.Rect(win, pos=[0, 0], width=0.35, height=0.5, fillColorSpace="rgb255",
                   lineColorSpace="rgb255")
rect.fillColor = rgbs[0]*255
rect.lineColor = rect.fillColor


rect2 = visual.Rect(win, pos=[0, 0.5], width=0.35, height=0.5, fillColorSpace="rgb",
                   lineColorSpace="rgb")
rect.fillColor = rgbs[1]*255
rect.lineColor = rect.fillColor

win.flip()
'''


'''
-> make sure to load that calibration file + test rgb to lms
'''

#c1 = Calibration()
#c1.save_to_file()

#c2 = Calibration()
#c2.load_from_file(path="cal_test")

'''
def M(l, m, R, phi):
    return 0.5*(m + l - R*np.cos(phi))


def L(l, m, R, phi):
    return 0.5*(m + l + R*np.cos(phi))


def M_y(l, m, R, phi):
    return m*(1 - R*np.cos(phi)/(1 + m/l))


def L_y(l, m, R, phi):
    return l*(1 + R*np.cos(phi)/(l/m + 1))


l = 0.5
m = 0.3
R = 0.05
phis = np.arange(0, 2*np.pi, 0.01)

fig, ax = pl.subplots(ncols=2)
ax[0].plot(phis, L(l, m, R, phis), c="r", label="L")
ax[0].plot(phis, M(l, m, R, phis), c="g", label="M")
ax[0].plot(phis, L(l, m, R, phis)-M(l, m, R, phis), c="b", label="L-M")

ax[1].plot(phis, L_y(l, m, R, phis), c="r", label="L")
ax[1].plot(phis, M_y(l, m, R, phis), c="g", label="M")
ax[1].plot(phis, L_y(l, m, R, phis)-M_y(l, m, R, phis), c="b", label="L-M")
pl.legend()
pl.show()
'''
'''
#draw the stimuli and update the window
while True: #this creates a never-ending loop
    grating.setPhase(0.05, '+')#advance phase by 0.05 of a cycle
    grating.draw()
    fixation.draw()
    mywin.flip()

    if len(event.getKeys())>0:
        break
    event.clearEvents()
'''
'''
s = Subject(short="jdoe", name="Doe", surname="John", birthday="01.01.81")
print(s.uuid)
s.save_to_file()

s2 = Subject()
s2.load_from_file("subject_jdoe.json")
print(s2.name, s2.surname, s2.birthday, s2.uuid)
#'''