"""
Functions independent of any module.
"""

import os
import json
import codecs
import ruamel.yaml
import numpy as np


def float2hex(x):
    """
    Convert float to HEX value.
    """
    val = hex(int(round(x)))[2:]
    val = "0" + val if len(val) < 2 else val
    return val


def prepare_for_yaml(d, list_compression=True):
    """
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


def dump_file(dt, path, filetype, list_compression=True):
    """
    Dump a dictionary to yaml/json file.

    :param dt: Dictionary that should be saved.
    :param path: File path.
    :param filetype: Filetype, either json or yaml. Files get saved accordingly.
    :param list_compression: If True, lists will be written with brackets. Default is True.
    :return: True.
    """
    save_dir, save_file = os.path.split(path)
    if save_dir and not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if filetype.lower() == "yaml" or filetype.lower() == "yml":
        dt = prepare_for_yaml(dt, list_compression=list_compression)
        dt = dict(sorted(dt.items()))
        if ".yaml" not in save_file:
            path = path + ".yaml"
        elif filetype == "yml" and ".yml" not in save_file:
            path = path + ".yml"
        with open(path, "w") as outfile:
            ruamel.yaml.YAML().dump(dt, outfile)
    elif filetype.lower() == "json":
        if ".json" not in save_file:
            path = path + ".json"
        json.dump(dt, codecs.open(path, 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True, indent=4)
    return True


def sine_fitter(x, amp, phi, off):
    """
    For iso-slant fit.
    :param x: Hue angle.
    :param amp: Amplitude.
    :param phi: Phase.
    :param off: Offset.
    :return: Sine value.
    """
    return amp * np.sin(x + phi) + off

