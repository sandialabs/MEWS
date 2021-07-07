from setuptools import setup, find_packages
import os
import re
import numpy as np
from distutils.core import setup
from distutils.extension import Extension



DISTNAME = 'mews'
PACKAGES = find_packages()
DESCRIPTION = 'Multi-scenario Extreme Weather Simulator'
AUTHOR = 'MEWS Developers'
MAINTAINER_EMAIL = 'dlvilla@sandia.gov'
LICENSE = 'Revised BSD'
URL = 'None'
DEPENDENCIES = ['numpy', 'pandas', 'matplotlib','scipy']

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("mews.cython.markov", ["mews/cython/markov.pyx"]),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [
        Extension("mews.cython.markov", ["mews/cython/markov.pyx"]),
    ]

# use README file as the long description
file_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(file_dir, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# get version from __init__.py
with open(os.path.join(file_dir, 'mews', '__init__.py')) as f:
    version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        VERSION = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")
        

setup(name=DISTNAME,
      cmdclass=cmdclass,
      version=VERSION,
      packages=PACKAGES,
      ext_modules=ext_modules,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      zip_safe=False,
      install_requires=DEPENDENCIES,
      scripts=[],
      include_package_data=True,
      include_dirs = [np.get_include()])

