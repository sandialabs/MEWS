.. MEWS documentation master file, created by
   sphinx-quickstart on Tue Jun 21 08:08:36 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MEWS Documentation
===================

.. figure:: images/logo.png

The Multi-scenario Extreme Weather Simulator (MEWS) is a Python package that projects the increased frequency
and severity of extreme temperature events. A simplified kinematic model of Hurricanes has been created  

The software has one major use case that is supported by the run_mews module. After installation (see the MEWS GitHub repository (https://github.com/sandialabs/MEWS), one can use the extreme_temperature function to create new analyses or to generate EnergyPlus files for existing ones.

The ability to do an EnergyPlus study has also been added.

===========
Quick Start
===========

To start using MEWS, open a python console or IDE like Spyder and import the extreme_temperature function::

     from mews.run_mews import extreme_temperature

You can use the example in `getting started example <https://github.com/sandialabs/MEWS/blob/examples/run_mews_extreme_temperature_example_v_1_1_2.py>`, that has descriptions of basic inputs for getting started. The code can be used to either 1) generate EnergyPlus files from precalculated solutions to the MEWS problem contained in the repository or 2) Running a new model fit optimization. The first option only requires several hours to run whereas the second option requires several days on several dozen processors depending on the optimization parameter inputs.


```

.. toctree::
   :maxdepth: 1
   
   overview
   modules
   license
   
   
   
