"""Import helpers: pipeline scripts have digit-prefixed names."""
import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def load_script(name):
    path = os.path.join(ROOT, "scripts", name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
