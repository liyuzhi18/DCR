# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:44:14 2025

@author: Enac Gallardo-Diaz

Electron Impact Excitation 
Calculate rate coefficients from cross sections

1) Initializations and input arguments

2) Read input files

3) Construct and output X-section (optional))

4) Calculate rate coefficients using Gauss Quadrature integration

5) Output data

6) Plot Rate Coefficient (optional))


Input parameters by the user are clearly signaled with ***********************

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from Maxwellian_integration import Rate_Coefficient
from constants import *
from XSec_output import XSec_Output
from find_names import *
from fit_func_str2py import fit_func_str2py


## 1) INITIALIZATIONS ----------------------------------------------------------
     
## 1.1) Setup temperature grid
# In Kelvin
# Choose LINEAR or LOG grid by commenting either

# LINEAR
# Temp_K_min = 1000 # *********************************************************
# Temp_K_max = 10000 # ********************************************************
# Numb_Temp = 10 # number of Temp points in the T grid # **********************
# Temp_K_Grid=np.linspace(Temp_K_min,Temp_K_max,Numb_Temp)

# LOG
# Choose exponents to create a log grid
Temp_K_exp_min = 0 # **********************************************************
Temp_K_exp_max = 2 # **********************************************************
Numb_Temp = 50 # number of Temp points in the T grid # ************************
Temp_K_Grid=np.logspace(Temp_K_exp_min,Temp_K_exp_max,Numb_Temp)

Temp_eV_Grid=Temp_K_Grid*K2eV

## 1.2) Energy range specifications for integral
# in eV
# The minimum value of the energy grid will be the Threshold Energy of transition
E_grid_max=1000.0 # ***********************************************************
# Tolerance for Gaussian Quadrature integration method
tol=1e-25 # *******************************************************************

## 1.3) Input/Output Files
path_output=r'/Users/asimida/Desktop/lanl_work/codes/EIE_results/'
# Choose all vi states from input folder
initial_state='X1Sg' # ********************************************************
path_fits=r'/Users/asimida/Desktop/lanl_work/codes/EIE_Xsec2RC/'+ initial_state +'/fits/' # ***
folder_names_vi = find_folder_names(path_fits)
n_vi=np.shape(folder_names_vi)[0]
vi=string_integers = [str(i) for i in range(n_vi)]

Total_EIE_RC=np.zeros((len(vi),Numb_Temp))

pd.set_option('display.max_colwidth', None) # Allows to convert long cells into strings without truncating (for reading Fit Funct) 

# Run through all selected vi states
n=0 
while n<len(vi):
    path_input=path_fits+'/vi='+vi[n] 
    
    # Find the names of the files for each transition in the database input path
    file_name_trans = find_file_names(path_input)
    n_trans=np.shape(file_name_trans)[0]
    
    # Loop through each transition
    t=0
    while t<n_trans : 
        input_file=file_name_trans[t]
        EIE_Output_File=input_file[0:-7]+'Tot_RC_Gauss.out'
        ICS_Output_File=input_file[0:-7]+'X_sec_vf='
        
## 2) READ INPUT ---------------------------------------------------------------
        fullfile=os.path.join(path_input, input_file)
    
        data_fit_func=pd.read_csv(fullfile, skip_blank_lines=True, header=None, skiprows=6, nrows=1, dtype=str,encoding_errors='ignore') # Read Fit Function from file
        data_file=pd.read_csv(fullfile,delim_whitespace=True, header=8, skip_blank_lines=True, skiprows=1)
        v_dim=np.shape(data_file)[0]
        n_fit_params=np.shape(data_file)[1] - 5 # Number of parameters in Fit function

        # Read Fit Function 
        fit_func_str=str(data_fit_func.at[0,0])[20:] # Read Fit Function from file
        fit_func_str_py=fit_func_str2py(fit_func_str) # Convert into python syntax
        
        # Memory allocation
        E_th =  np.zeros((v_dim))
        a = np.zeros((v_dim,n_fit_params))
        EIE_RC=np.zeros((v_dim,Numb_Temp))
        EIE_RC_error=np.zeros((v_dim,Numb_Temp))
    
        # Print reading file
        print("File:",input_file,"Rows=",data_file.shape[0],"Cols=",data_file.shape[1]) # Number of rows (lines) and columns in the file
        print('Fit Function',fit_func_str)
        print('vf  E Threshold (eV) [ a0  a1 ... an ]')
    
        # Run through all vf states
        v=0
        while v<v_dim:
            E_th[v] = data_file.iloc[v,3]
            w=0
            while w<n_fit_params:
                a[v,w] = data_file.iloc[v,w+4]
                w+=1
                        
            print(v, E_th[v], a[v,:])
        
            # Define the fit function:
            exec(fit_func_str_py)    
                    
    
## 3) CONSTRUCT AND OUTPUT X-SEC (optional)  ----------------------------------
    
            Egrid_Npts=10000 # Number of Energy points for X-sec 
            XSec_Output(E_th[v], E_grid_max, Egrid_Npts, Fit, ICS_Output_File, path_output, v)
            
        
## 4) CALCULATE RATE COEFFICIENTS ---------------------------------------------
            # Run through Temperature grid
            i = 0
            while i<Numb_Temp: 
                # EIE Rate Ceofficient
                my=Rate_Coefficient(Temp_eV=Temp_eV_Grid[i], FIT=Fit, E_th=E_th[v], E_grid_max=E_grid_max, tol=tol)
                RC_value_error=my.Calculate_Rate_Coefficient_Gauss()
                EIE_RC[v,i]=RC_value_error[0]
                EIE_RC_error[v,i]=RC_value_error[1]            
                i += 1
            v+=1
    

## 5) OUTPUT DATA -------------------------------------------------------------
        Output_File = open(os.path.join(path_output,EIE_Output_File,),'w')
        Output_File.write("# Temperature (K), Total Electron Impact Excitation RC (cm3 s-1) \n")
        Total_EIE_RC[n][:]=np.sum(EIE_RC,axis=0)
        i=0
        while i<Numb_Temp: # Temperature Grid
            print(Temp_K_Grid[i], Total_EIE_RC[n][i], file=Output_File)
            i+=1
        Output_File.close()
        
        t+=1
    n+=1



## Evaluate the total rate coefficients for different vi states

## 6) PLOT RATE COEFFICIENTS (optional) ---------------------------------------    



# plt.plot(Temp_K_Grid/1e3,Total_EIE_RC[0],'k',linestyle='dotted',label='vi=0')
# plt.plot(Temp_K_Grid/1e3,Total_EIE_RC[1],'g',linestyle='dotted',label='vi=1')
# plt.plot(Temp_K_Grid/1e3,Total_EIE_RC[2],'b',linestyle='dotted',label='vi=2')
# plt.plot(Temp_K_Grid/1e3,Total_EIE_RC[3],'r',linestyle='dotted',label='vi=3')
# plt.xlabel("Temperature (1e3 K)")
# plt.ylabel("Rate Coefficient (cc/s)")
# plt.legend()
# plt.ylim(1e-15,1e-7)
# plt.xlim(5,300)
# plt.xscale('log')  # Set y-axis to log scale
# plt.yscale('log')  # Set y-axis to log scale
# plt.show()








