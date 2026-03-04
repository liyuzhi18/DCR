#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:20:34 2023

@author: uchihali
"""
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("quad_max.in",usecols=range(1,3))
eng = data[:,0]
f = data[:,1]
plt.plot(eng,f)