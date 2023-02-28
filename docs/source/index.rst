.. MEWS documentation master file, created by
   sphinx-quickstart on Tue Jun 21 08:08:36 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MEWS Documentation
===================

.. figure:: images/logo.png

The Multi-scenario Extreme Weather Simulator (MEWS) is a python-based software in development to predict the increased frequency
and severity of various natural weather phenomena including: heat waves and cold snaps. A simplified kinematic model of Hurricanes is coming in the next version. 

The software has one major use case that is supported by the run_mews module. After installation (see the MEWS GitHub repository (https://github.com/sandialabs/MEWS), one can use the extreme_temperature function to create new analyses or to generate EnergyPlus files for existing ones.

```
from mews.run_mews import extreme_temperature




```

.. toctree::
   :maxdepth: 1
   
   overview
   modules
   license
   
   
   
