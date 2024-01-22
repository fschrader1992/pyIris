"""
SCRIPT WITH FUNCTIONS FROM NEWER PYIRIS VERSION.
CAN BE USED TO CONVERT SPECTRUM FILES WITH OLDER PYTHON/NIX VERSIONS.

Run this on an older version before upgrading.
"""

import os
import argparse
import numpy as np
import ruamel.yaml

from pyiris.spectrum import Spectrum


def prepare_for_yaml(d, list_compression=True):
    """
    Imported from pyiris.functions.

    Handle dictionary to save as yaml.

    :param d: Dictionary.
    :param list_compression: If True, lists will be written with brackets. Default is True.
    :return: Cleaned dictionary.
    """
    for k, v in d.items():
        # replace numpy arrays etc.
        if isinstance(v, dict):
            v = prepare_for_yaml(v, list_compression=list_compression)
        if isinstance(v, np.ndarray):
            v = v.astype(float).tolist()
        if isinstance(v, np.float64):
            v = v.astype(float)
        if isinstance(v, list):
            v = list(map(lambda vl: vl if not isinstance(vl, np.ndarray) else vl.tolist(), v))
            v = list(map(lambda vl: vl if not type(vl).__module__ == np.__name__ else str(vl), v))
            # set the yaml flow style to brackets to reduce number of lines in datafile
            if list_compression:
                v = ruamel.yaml.comments.CommentedSeq(v)
                v.fa.set_flow_style()
        else:
            if type(v).__module__ == np.__name__:
                v = float(v)
        d[k] = v
    return d


def dump_old_yaml_file(dt, path, list_compression=True):
    """
    Modified from pyiris.functions.

    Dump a dictionary to yaml/json file.

    :param dt: Dictionary that should be saved.
    :param path: File path.
    :param list_compression: If True, lists will be written with brackets. Default is True.
    :return: True.
    """
    dt = prepare_for_yaml(dt, list_compression=list_compression)
    dt = dict(sorted(dt.items()))
    if ".yaml" not in path:
        path = path + ".yaml"
    with open(path, "w") as outfile:
        ruamel.yaml.YAML().dump(dt, outfile)
    return True


def save_old_nix_as_yaml(spec, path, list_compression=True):
    """
    Slightly modified from pyiris.spectrum.Spectrum.
    Save spectrum data to yaml file.
    :param path: Location of file.
    """

    dt = dict({})
    md = dict({})

    # add metadata
    md.update(vars(spec))
    md["date"] = spec.date.isoformat()
    md["uuid"] = str(spec.uuid)
    try:
        md["photometer"] = int(spec.photometer.getDeviceSN())
    except AttributeError:
        md["photometer"] = 0
    del md["monitor"]
    del md["path"]
    del md["spectra"]

    # sort keys
    dt["metadata"] = dict(sorted(md.items()))

    # add spectral data
    # create dict with 2 layers from tuple keys to make file more readable
    sd = dict.fromkeys(spec.names, None)
    for sk, sv in spec.spectra.items():
        if sd[sk[0]] is None:
            sd[sk[0]] = dict({})
        if sk[1] in ["tristim", "uv", "xy", "colortemp"]:
            sv = list(map(lambda v: str(v).replace("\n", "").replace("\r", ""), sv))
        sd[sk[0]][sk[1]] = sv

    dt.update(sd)
    # process trial data
    dt = prepare_for_yaml(dt, list_compression=list_compression)

    # write datafile
    with open(path, "w") as outfile:
        ruamel.yaml.YAML().dump(dt, outfile)

    print("Successfully saved spectra to yaml-file {}".format(path))
    return True


if __name__ == "__main__":
    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description="Convert nix files created with older pyiris versions to yaml.")
    parser.add_argument("-d", "--directory", metavar="", type=str, default="",
                        help='Directory in which all nix files should be replaced.')
    parser.add_argument("-p", "--path", metavar="", type=str, default="",
                        help="Path to nix file.")
    parser.add_argument("-l", "--listcompression", metavar="", type=bool, default=True,
                        help="If True, lists will be written with brackets. Default is True.")
    args = parser.parse_args()

    nix_files = []

    if len(args.directory) > 0:
        for root, dirs, files in os.walk(args.directory):
            for file in files:
                if file.endswith(".nix"):
                    nix_files += [os.path.join(root, file)]
    else:
        nix_files = [args.path]

    for nix_path in nix_files:
        spec = Spectrum()
        spec.load_from_file(nix_path)

        save_old_nix_as_yaml(spec, path=nix_path.replace(".nix", ".yaml"), list_compression=args.listcompression)
