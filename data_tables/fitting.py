#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data fitting for H2 vibrational excitation
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# candidate functions
def func1(x, A1, A2, A3, A4, A5, A6):
    return (A1*np.log(x) + A2+A3/x+A4/x**2+A5/x**3)*((x+1)/(x+A6))

def func2(x, A1, A2, A3, A4, A5):
    return (A1+A2/x+A3/x**2+A4/x**3)*(x**2/(x+A5))

# reading csv file
df = pd.read_csv("people.csv")