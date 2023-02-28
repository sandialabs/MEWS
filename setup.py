from setuptools import setup, find_packages, Extension
import os
import re
import numpy as np

DISTNAME = 'mews'
PACKAGES = find_packages()
DESCRIPTION = 'Multi-scenario Extreme Weather Simulator (MEWS)'
AUTHOR = 'MEWS Developers'
MAINTAINER_EMAIL = 'dlvilla@sandia.gov'
LICENSE = 'Revised BSD'
URL = 'https://github.com/sandialabs/MEWS'
DEPENDENCIES = ['numpy', 'pandas', 'matplotlib','scipy','statsmodels',
                'openpyxl', 'cython',
                'xarray[complete]']

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    # include_dirs is needed here for MAC OS systems
    ext_modules += [
        Extension("mews.cython.markov",
                  ["mews/cython/markov.pyx"],
                  include_dirs=[np.get_include()]),
                  #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
                  # This define macros gets rid of an error but then causes the cython code
                  # to not accept .shape[0] type calls on arrays. https://stackoverflow.com/questions/35414980/cython-dimensions-is-not-a-member-of-tagpyarrayobject
        Extension("mews.cython.markov_time_dependent",
                  ["mews/cython/markov_time_dependent.pyx"],
                  include_dirs=[np.get_include()])]
                  #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),]
    
    cmdclass.update({'build_ext': build_ext})
else:
    pass
    # ext_modules += [
    #     Extension("mews.cython.markov",
    #               ["mews/cython/markov.pyx"],
    #               include_dirs=[np.get_include()],
    #               define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
    #     Extension("mews.cython.markov_time_dependent",
    #               ["mews/cython/markov_time_dependent.pyx"],
    #               include_dirs=[np.get_include()],
    #               define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
    # ]

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
      long_description_content_type="text/markdown",
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=MAINTAINER_EMAIL,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      download_url='https://github.com/sandialabs/MEWS/archive/refs/tags/v0.1.0.tar.gz',
      zip_safe=False,
      install_requires=DEPENDENCIES,
      scripts=[],
      include_package_data=True,
      include_dirs = [np.get_include()],
      keywords = ["Building Energy Modeling","Infrastructure","Extreme Weather","Markov","Resilience","Energy Plus","DOE2","DOE-2"])

