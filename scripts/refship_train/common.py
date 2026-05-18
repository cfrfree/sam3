import importlib
import os
import sys


def _load_rrsis_utils():
    rrsis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "rrsis"))
    if rrsis_dir not in sys.path:
        sys.path.append(rrsis_dir)
    return importlib.import_module("utils")


utils = _load_rrsis_utils()
