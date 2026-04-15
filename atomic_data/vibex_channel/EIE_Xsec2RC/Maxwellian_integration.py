"""
Python Maxwellian Function codes
Author:  Mark Zammit, T-1, Los Alamos National Laboratory
Date:  9/2/21

This code follows the methods of: Y.-K. Kim and M. E. Rudd, Phys. Rev. A 50, 3954 (1994)

See NO_X.py for usage example

Modification: 2/3/2025
Author: Enac Gallardo-Diaz
Include Gauss quadrature method of integration

"""

from __future__ import print_function
import numpy as np
from scipy.integrate import quad

Ryd = 13.6057  # eV. 1 Rydberg = 13.6057 eV


class Rate_Coefficient:

    """
    Class to set up Energy Grid Arrays 
    base class contains _required and _allowed keys
    _required: keys that must be supplied by user
    _allowed: extra optional keys

    _required:
        Temp_eV: Temperature (eV)
        FIT: fit function of the X-section
        E_th: Thershold Energy (eV) (also the lower bound of integration)
        E_grid_max: upper bound of integration (eV)

        
    """

    _required = ['Temp_eV','FIT','E_th','E_grid_max']
    _allowed = _required+['tol']


    def __init__(self,**kwargs):
        """
        set up Incident Energy Grid
        :param kwargs: dictionary of _required or _allowed items
        """

        # check all _required are present
        for key in self._required:
            assert key in kwargs.keys()
            
        # check that item is in _allowed and set
        for k,v in kwargs.items():
            assert k in self._allowed
            setattr(self,k,v)


    def Calculate_Rate_Coefficient_Gauss(self):
        """
        Calculates the rate coefficient for a Maxwellian function using Gauss quadrature.
        This is different than trapz because needs to define a function integrand
        :return: , 
        """        
        
        # Maxwellian distribution function for a given Temp_eV and incident energy Ei
        def F_Maxwellian_func(Ei):
             k_BT = self.Temp_eV
             return (2.0/k_BT)  * ( (Ei/(np.pi*k_BT) )**(0.5)) * np.exp( -Ei / k_BT )
        
        # Velocity for a given incident energy Ei
        def v_func(Ei):
            eV_erg = 1.0/6.2415E11 # 1 erg (g cm2 / s2) = 6.2415 x 10^11 eV
            me_g = 9.10938E-28 # electron mass in grams
            constant = eV_erg / me_g
            return ( 2.0*Ei*constant )**(0.5) # cm / s
        
        def integrand(Ei): # v * F_maxwellian * Fit_funct
            return v_func(Ei) * F_Maxwellian_func(Ei) * self.FIT(Ei/self.E_th) # FIT is defined for x=Ei/E0
        
        rate_coeff = quad(integrand, self.E_th,self.E_grid_max,epsabs=self.tol,epsrel=self.tol)
      
        return rate_coeff
    
