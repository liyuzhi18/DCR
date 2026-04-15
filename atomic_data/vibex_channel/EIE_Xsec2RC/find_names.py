#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:33:20 2025

@author: enac
Create a list of files name strings in a given path
"""

import os
import numpy as np

def find_file_names(folder_path):
  """Lists all files in the given directory.

  Args:
    folder_path: The path to the directory.

  Returns:
    A list of strings, where each string is the name of a file in the directory.
    Returns an empty list if the directory does not exist or is empty.
  """
  try:
    #files = os.listdir(folder_path)
    # Skip .DS_Store and hidden files
    file_names = [f for f in os.listdir(folder_path) 
                  if not f.startswith('.') 
                  and os.path.isfile(os.path.join(folder_path, f))]
    return np.array(file_names)
    #return [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
  except FileNotFoundError:
    return []

def find_folder_names(folder_path):
  """Lists all folders in the given directory.

  Args:
    folder_path: The path to the directory.

  Returns:
    A list of strings, where each string is the name of a folder in the directory.
    Returns an empty list if the directory does not exist or is empty.
  """
  try:
    files = os.listdir(folder_path)
    return [r for r in files if os.path.isdir(os.path.join(folder_path, r))]
  except FileNotFoundError:
    return []

def find_levels_in_filename(filename, known_levels):
    matches = []
    for level in known_levels:
        pos = filename.find(level)
        if pos != -1:
            matches.append((level, pos))

    if len(matches) != 2:
        raise ValueError(f"Expected 2 level matches, found {len(matches)} in {filename}: {matches}")
    
    # Sort by appearance order in filename
    matches.sort(key=lambda x: x[1])
    
    upper_level = matches[0][0]
    lower_level = matches[1][0]
    
    return upper_level, lower_level
