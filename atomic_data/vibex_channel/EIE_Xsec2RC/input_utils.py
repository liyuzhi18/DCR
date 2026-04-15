#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:17:36 2025

@author: asimida
"""
import os
import re
import glob
import pandas as pd
import numpy as np
from Maxwellian_integration import Rate_Coefficient
from constants import *

def get_struct(base_directory):
    data_stuct = {}
    folder_pattern = re.compile(r"vi\s*=\s*(\d+)")
    # Get all directories in the base directory
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        
        if os.path.isdir(folder_path):
            match = folder_pattern.search(folder_name)
            
        if match:
            vib_state = str(match.group(1))
        data_files = glob.glob(os.path.join(folder_path,"*.txt"))
        
        if vib_state not in data_stuct:
            data_stuct[vib_state] = {}
            
        for file in data_files:
            file_name = os.path.basename(file)
        
            # Extract fnl_state (e.g., "B1Su" from the filename)
            fnl_state_match = re.search(r"MCCC-el-H2-([\w\d]+)\.X1Sg", file_name)
            if fnl_state_match:
                fnl_state = fnl_state_match.group(1)
                        
                data_stuct[vib_state][fnl_state] = {}

    
    return data_stuct

def fit_func_str2py(input_str):
  """
  Inputs the Fit Function Str from main.py and converts it into a str of a defined function Fit that python can read using exec

  Args:
    input_str: The string to modify.

  Returns:
    A new string with py sintax
  """
# Replace math operators/expressions
  output_str = input_str.replace("^", "**").replace("exp", "np.exp").replace("log", "np.log").replace("|", " ")
  # Replace parameter variables a0...a7 with params[index]
  for i in range(8):
     output_str = output_str.replace(f"a{i}", f"params[{i}]")
  fit_string = (
     "def fit_func(x, params):\n"
    f"    return abs({output_str})*bohr_cm*bohr_cm"
  )
  return fit_string
         
def get_file_path(base_directory, vib_state, final_state):
    folder_name = f"vi={vib_state}"
    file_name = f"MCCC-el-H2-{final_state}.X1Sg_vi={vib_state}_fit.txt"
    return os.path.join(base_directory, folder_name, file_name)

def populate_data_structure(data_struct, base_directory, temp_grid):
    """
    Populate the data structure with energy loss and rate coefficients.
    """
    # Process each vibrational state and final state
    for vib_state in data_struct:
        for final_state in data_struct[vib_state]:
            file_path = get_file_path(base_directory, vib_state, final_state)
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            
            # Parse the file and get the fit function
            formated_data = parse_file_contents(file_path)
            fit_func_str = formated_data["fit_func_str"]
                                       
            # Process each vf
            for vf, vf_data in formated_data["vf"].items():
                eng = vf_data["eng"]
                    
                # Handle non-numeric vf values
                try:
                    vf_float = float(vf)
                except ValueError:
                    #print(f"Warning: Non-numeric vf value '{vf}'. Skipping.")
                    continue
                    
                print(f"Processing {vib_state}/{final_state}/{vf} with eng={eng}")
                    
                # Calculate rate coefficients
                rate_coefs = calculate_rate_coefficient(
                    formated_data["fit_func_str"], 
                    vf_float,
                    vf_data["fit_params"],
                    eng,
                    temp_grid
                )
                    
                    # Update data structure
                data_struct[vib_state][final_state][vf] = {
                    "energy_loss": eng,
                    "rate_coefs": rate_coefs
                }
                    
    return data_struct

def calculate_rate_coefficient(fit_func_str, vf, params, eng, temp_grid):
    """
    Calculate rate coefficients by integrating cross-sections over Maxwellian distributions.
    """
    from scipy.integrate import quad
    import numpy as np
    
    # Define fit_func in this function's scope
    local_namespace = {'np': np,'bohr_cm': bohr_cm}
    exec(fit_func_str, local_namespace)
    fit_func = local_namespace['fit_func']
    
    # Convert to numpy array
    temp_array = np.asarray(temp_grid)
    n_temps = len(temp_array)
    
    # Pre-allocate results
    rate_coefs = np.zeros(n_temps)
    
    # Define constants
    e_th = eng  # Threshold energy in eV
    e_max = 10.0 * e_th if e_th > 0 else 100.0  # Maximum energy for integration
    tol = 1.0e-6  # Integration tolerance
    
    # Constants for velocity calculation
    eV_erg = 1.0/6.2415E11  # 1 erg = 6.2415 x 10^11 eV
    me_g = 9.10938E-28      # electron mass in grams
    velocity_const = np.sqrt(2.0 * eV_erg / me_g)  # For v(E) = sqrt(2E/m)
    
    # create the fitting function for calculating the cross section
    local_namespace = {}
    n_params = np.shape(params)
    
    # Process each temperature
    for i, temp_eV in enumerate(temp_array):
        # Define the integrand: v * F_maxwellian * cross_section
        
        def integrand(energy):
            # Velocity calculation
            velocity = velocity_const * np.sqrt(energy)  # cm/s
            
            # Maxwellian distribution
            maxwellian = (2.0/temp_eV) * np.sqrt(energy/(np.pi*temp_eV)) * np.exp(-energy/temp_eV)
            
            # Cross section - add error checking
            try:
                # Important: The fit function might need the energy, not just vf
                # Try both approaches
                try:
                    # Try with energy/e_th (normalized energy)
                    x = energy / e_th if e_th > 0 else energy
                    cross_section = fit_func(x, params)
                except:
                    # If that fails, try with just vf
                    cross_section = fit_func(vf, params)
                
                if cross_section is None:
                    cross_section = 1.0
            except Exception as e:
                print(f"Error calculating cross section at E={energy:.2f} eV: {e}")
                cross_section = 1.0
            
            # Complete integrand
            return velocity * maxwellian * cross_section
        
        # Perform the integration
        try:
            result, error = quad(integrand, e_th, e_max, epsabs=tol, epsrel=tol)
            rate_coefs[i] = result
        except Exception as e:
            print(f"Integration error at T={temp_eV} eV: {e}")
            rate_coefs[i] = 0.0
    
    return rate_coefs


def parse_file_contents(abs_filepath):
    data_fit_func = pd.read_csv(abs_filepath, skip_blank_lines=True, header=None, skiprows=6, nrows=1, dtype=str, encoding_errors='ignore')
    data_file = pd.read_csv(abs_filepath, delim_whitespace=True, header=8, skip_blank_lines=True, skiprows=1)
    n_fit_params = np.shape(data_file)[1] - 5
    
    # Read Fit Function
    fit_func_str = str(data_fit_func.at[0,0])[20:]
    fit_func_str_py = fit_func_str2py(fit_func_str)
    
    # Get all the data
    vfs = data_file.iloc[:,0].tolist()
    engs = data_file.iloc[:,3].tolist()
    all_params = data_file.iloc[:,4:4+n_fit_params]
    
    # Create the restructured data format
    vf_data = {}
    for i, vf in enumerate(vfs):
        vf_str = str(vf)  # Convert to string for dictionary key
        params = all_params.iloc[i].tolist()  # Get parameters for this vf
        
        vf_data[vf_str] = {
            "eng": engs[i],
            "fit_params": params
        }
    
    file_data = {
        "fit_func_str": fit_func_str_py,
        "vf": vf_data
    }
    
    return file_data


def cal_eng_loss(data_structure):
    eng_loss_per_vib = {}  # to hold results per vib_state
    rates = {}
    for vib_state in data_structure:
        energy_loss_rate = 0
        total_rate = 0
        for final_state in data_structure[vib_state]:
            for final_vib in data_structure[vib_state][final_state]:
                eng_loss_state = (
                    data_structure[vib_state][final_state][final_vib]['energy_loss']
                    * data_structure[vib_state][final_state][final_vib]['rate_coefs']
                )
                energy_loss_rate += eng_loss_state
                total_rate += data_structure[vib_state][final_state][final_vib]['rate_coefs']
                
            eng_loss_per_vib[vib_state] = energy_loss_rate / total_rate
            rates[vib_state] = total_rate
    
    return eng_loss_per_vib, rates
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
