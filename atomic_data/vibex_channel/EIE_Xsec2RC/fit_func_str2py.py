#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:19:00 2025

@author: enac
"""

def fit_func_str2py(input_str):
  """
  Inputs the Fit Function Str from main.py and converts it into a str of a defined function Fit that python can read using exec

  Args:
    input_str: The string to modify.

  Returns:
    A new string with py sintax
  """
  # Replace Syntax
  output_str=input_str.replace("^", "**")
  output_str=output_str.replace("exp", "np.exp")
  output_str=output_str.replace("log", "np.log")
  output_str=output_str.replace("|", " ")
  # REplace variable names 
  output_str=output_str.replace("a0", "a[v,0]")
  output_str=output_str.replace("a1", "a[v,1]")
  output_str=output_str.replace("a2", "a[v,2]")
  output_str=output_str.replace("a3", "a[v,3]")
  output_str=output_str.replace("a4", "a[v,4]")
  output_str=output_str.replace("a5", "a[v,5]")
  output_str=output_str.replace("a6", "a[v,6]")
  output_str=output_str.replace("a7", "a[v,7]")
  
  Fit_string = """
def Fit (x):
  return abs(""" + output_str + """) * bohr_cm * bohr_cm
"""

  return Fit_string