# Version 1.1.3

 - Added shift_function to mews.weather.alter.Alter which shifts an entire
   time series mean value by "m" while keeping the timeseries minimum value constant
 - Added recalculate_psychrometrics to the "mews.weather.alter.Alter" class so that changes
   to a psychrometric variable shift other psychrometric variables. 
 - Added EPW_DATA_DICTIONARY to mews.constants.data_format and several other 
   improvements to assure EPW maintain the correct data format.
 - Added changes to mews.epw.epw focused on assuring that correct EPW files
   are written so that any other program can read them.
 - Modernized code for Python 3.13.2

# Version 1.1.2

- Added helper scripts run_mews and run_energyplus to make completing studies more easy.
- Updated to run in Python 3.9 - 3.12
- Added ground temperature changes into the future in weather files.
- This was the beginning of the change-log. Many other minor changes have been made
  but the main algorithm has not been altered.

# Version 1.1.1

- NO CHANGELOG KEPT.