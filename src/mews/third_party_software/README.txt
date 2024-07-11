The executables posted are only work on Windows. 

These executables will be eliminated once the native python BIN2TXT and TXT2BIN equivalent functions 

have been well tested. For other platforms, recompiling the Fortran code is required.

METHODS to replace BIN2TXT.EXE and TXT2BIN.EXE:
          mews.weather.DOE2Weather.read_doe2_bin and  
          mews.weather.DOE2Weather.write_doe2_bin
 
These are translations of parts of BIN2TXT.F and TXT2BIN.F translated with permission
from James and Jeff Hirsch and Associates (JJH&A). THE LICENSE FOR THESE 
UTILITIES MUST BE FORMED WITH (JJH&A) BEFORE THEY ARE DISTRIBUTED IN ANY OTHER
PACKAGE BESIDES MEWS.

The original correspondence obtaining permission is provided below. A similar
request must be made for other packages:

        # Wed 7/7/2021 4:10 PM
        Yes, you have my permission to distribute your DOE-2 weather file 
        python libraries with TXT2BIN and BIN2TXT executables on Github or 
        also translate the fortran source versions of those apps into python 
        and then distribute on Github as long as your acknowledge the 
        JJH&A license
 
        ________________________________________
        Jeff Hirsch
        James J. Hirsch & Associates
        Voice mail: (XXX) XXX-XXXX
        mobile: (XXX) XXX-XXXX
        -----------------------------------------------------
        From: Villa, Daniel L 
        Sent: Wednesday, July 7, 2021 11:02 AM
        To: Jeff.Hirsch@DOE2.com 
        Subject: Tranlate TXT2BIN.F and BIN2TXT.F into Python and distribute 
                 as open source??
         
        Jeff,
         
        I have built a tool that inputs and outputs DOE-2 weather files with 
        extreme events. I use the TXT2BIN.F and BIN2TXT.F executables with 
        the version of DOE-2 that I have licensed with Hirsch and Associates. 
        I would like to be able to distribute the python libraries with these 
        executables on Github or else be able to translate the *.F files into 
        python but know that the code is distributed under your license. 
         
        Would Hirsch and Associates be willing to let me create Python 
        versions of TXT2BIN.F and BIN2TXT.F and to distribute them as open 
        source code on GitHUB? I understand and respect Hirsch and Associate’s 
        decision if this is not allowed. Thank you.
         
        Daniel Villa
        Energy-Water System Integration Department 08825
        Sandia National Laboratories
        dlvilla@sandia.gov
        XXX-XXX-XXXX