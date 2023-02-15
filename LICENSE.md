Copyright Notice
=================

Copyright 2023 National Technology and Engineering Solutions of Sandia, LLC. 
Under the terms of Contract DE-NA0003525, there is a non-exclusive license 
for use of this work by or on behalf of the U.S. Government. 
Export of this program may require a license from the 
United States Government.

License Notice
=================

This software is distributed under the Revised BSD License (see below).
MEWS also leverages a variety of third-party software packages, which
have separate licensing policies.  

Revised BSD License
-------------------

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright notice, this 
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.
* Neither the name of Sandia National Laboratories, nor the names of
  its contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Third-Party Libraries
=================================
MEWS includes the source code from the following:


1 ) EPW - Energy Plus Weather 
============================

EPW is not the EPW downloaded from pypi.org, it must be downloaded
from https://github.com/building-energy/epw

EPW License
===========

MIT License

Copyright (c) 2019 Building Energy Research Group (BERG)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

2 ) DOE2 BIN2TXT.F and TXT2BIN.F Translations:
=============================================

METHODS   mews.weather.DOE2Weather.read_doe2_bin and  
          mews.weather.DOE2Weather.write_doe2_bin



Are translations of parts of BIN2TXT.F and TXT2BIN.F translated with permission
from James and Jeff Hirsch and Associates (JJH&A). THE LICENSE FOR THESE 
UTILITIES MUST BE FORMED WITH (JJH&A) BEFORE THEY ARE DISTRIBUTED IN ANY OTHER
PACKAGE BESIDES MEWS.
 
The original correspondence is provided 
     # below:
     
 Wed 7/7/2021 4:10 PM
 Yes, you have my permission to distribute your DOE-2 weather file 
 python libraries with TXT2BIN and BIN2TXT executables on Github or 
 also translate the fortran source versions of those apps into python 
 and then distribute on Github as long as your acknowledge the 
 JJH&A license
 
 ________________________________________
 Jeff Hirsch
 James J. Hirsch & Associates

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
