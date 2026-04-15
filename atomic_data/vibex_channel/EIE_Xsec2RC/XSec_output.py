#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:47:14 2025

@author: Enac Gallardo-Diaz

Create Output File for Cross Sections for EIE_Xsec2RC_gauss/main.py 
"""

import numpy as np
import os

def XSec_Output(E_th,E_grid_max,Egrid_Npts,Fit,ICS_Out_File,path_output,v):
    
        E_inc=np.linspace(E_th,E_grid_max,Egrid_Npts)
        iE=0
        ICS = np.zeros((Egrid_Npts))
        while iE < Egrid_Npts:
            x = E_inc[iE] / E_th
            ICS[iE] = Fit(x)
            iE+=1
        ICS=np.insert(ICS,0,0.0) # Start ICS from zero
        E_inc=np.insert(E_inc,0,0.0)
      
        # Print Cross Sections
        temp_file = ICS_Out_File + str(v) + ".out"; print(temp_file)
        temp_file_full=os.path.join(path_output, temp_file)
        Output_File = open(temp_file_full,'w')
        iE=0
        while iE < len(E_inc):
            print(E_inc[iE], ICS[iE], file=Output_File)
            iE+=1
        Output_File.close()