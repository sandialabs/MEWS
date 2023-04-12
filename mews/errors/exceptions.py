"""
Error Handling for MEWS.

----

Copyright Notice
=================

Copyright 2023 National Technology and Engineering Solutions of Sandia, LLC. 
Under the terms of Contract DE-NA0003525, there is a non-exclusive license 
for use of this work by or on behalf of the U.S. Government. 
Export of this program may require a license from the 
United States Government.

Please refer to the LICENSE.md file for a full description of the license
terms for MEWS. 

The license for MEWS is the Modified BSD License and copyright information
must be replicated in any derivative works that use the source code.

"""

class EPWMissingDataFromFile(Exception):
    pass

class EPWFileReadFailure(Exception):
    pass

class EPWRepeatDateError(Exception):
    pass

class MaxVsAvgDistributionError(Exception):
    pass

class ExtremesIntegrationError(Exception):
    pass

class MEWSInputTemplateError(ValueError):
    pass

class EnergyPlusRunFailed(Exception):
    pass