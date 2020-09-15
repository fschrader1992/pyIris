from io import open
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

packages = ['pyiris']

with open('README.md', encoding="utf8") as f:
    description_text = f.read()

install_req = ["psychopy", "nixio>=1.5.0b1", "numpy", "scipy",
               "matplotlib", "pandas", "symfit", "h5py", "pyyaml"]

setup(
    name='pyiris',
    description='pyIris model',
    author="Felix Schrader, Yannan Su (LMU Munich)",
    packages=packages,
    version="0.0.1",
    test_suite='test',
    install_requires=install_req,
    include_package_data=True,
    long_description=description_text,
    long_description_content_type="text/markdown",
    license="BSD",
    entry_points={"console_scripts": ["pyiris.spectrum=pyiris.commands:spectrum",
                                      "pyiris.calibrate=pyiris.commands:calibrate",
                                      "pyiris.calibration.plot=pyiris.commands:plot_calibration",
                                      "pyiris.colorspace.isoslant=pyiris.commands:measure_iso_slant",
                                      "pyiris.colorspace.colorcircle=pyiris.commands:color_circle",
                                      "pyiris.subject=pyiris.commands:subject",
                                      ]}
)
