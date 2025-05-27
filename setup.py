# NO META-DATA here except for dynamic parameters and cython
from setuptools import setup, find_packages, Extension
import os
import re
import numpy as np

cmdclass = {}
ext_modules = []

# include_dirs is needed here for MAC OS systems
ext_modules += [
    Extension(
        "mews.cython.markov",
        ["src/mews/cython/markov.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "mews.cython.markov_time_dependent",
        ["src/mews/cython/markov_time_dependent.pyx"],
        include_dirs=[np.get_include()],
    ),
]


def version_function():
    # get version from __init__.py
    file_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(file_dir, "src", "mews", "__init__.py")) as f:
        version_file = f.read()
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
        )
        if version_match:
            VERSION = version_match.group(1)
        else:
            raise RuntimeError("Unable to find version string.")

    return VERSION


setup(
    version=version_function(),
    ext_modules=ext_modules,
)
