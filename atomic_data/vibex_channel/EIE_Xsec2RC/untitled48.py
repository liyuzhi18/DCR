#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:37:16 2025

@author: asimida
"""
from input_utils import *
import numpy as np


initial_state='X1Sg' 
base_directory=r'/Users/asimida/Desktop/lanl_work/codes/EIE_Xsec2RC/'+ initial_state +'/fits/' 
data_structure = get_struct(base_directory)

# set temperature grid
Temp_K_exp_min = 4 
Temp_K_exp_max = 5 
Numb_Temp = 50
Temp_K_Grid=np.logspace(Temp_K_exp_min,Temp_K_exp_max,Numb_Temp)
Temp_eV_Grid=Temp_K_Grid*K2eV

data_structure = populate_data_structure(data_structure, base_directory, Temp_eV_Grid)

energy_loss_vib,total_rate = cal_eng_loss(data_structure)

#%% plot energy loss
import matplotlib.pyplot as plt

sorted_vib = sorted(energy_loss_vib, key=lambda v: int(v))
num_vib = len(sorted_vib)

# Use a color map to generate as many distinct colors as you have vib states
cmap = plt.get_cmap('viridis')
colors = [cmap(i/num_vib) for i in range(num_vib)]

plt.figure(figsize=(9, 5))

for i, vib_state in enumerate(sorted_vib):
    eng_loss = energy_loss_vib[vib_state]
    plt.plot(Temp_eV_Grid, eng_loss, label=f'v={vib_state}', color=colors[i])

plt.xlabel('Temperature (eV)')
plt.ylabel('Average Energy Loss (eV)')
plt.title('Energy Loss for ' + r'$e + H_2(X^1\Sigma_g^+, v) \to e + H_2(*)$')
plt.legend(title='Vib state', ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(2)
for i, vib_state in enumerate(sorted_vib):
    rate = total_rate[vib_state]
    plt.semilogy(Temp_eV_Grid, rate, label=f'v={vib_state}', color=colors[i])


plt.xlabel('Temperature (eV)')
plt.ylabel('Rate')
plt.title(r'$e + H_2(X^1\Sigma_g^+, v) \to e + H_2(*)$')
plt.legend(title='Vib state', ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
