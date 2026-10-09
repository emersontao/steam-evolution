# constants and functions for the Demo notebook

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import seaborn as sns
import matplotlib

## astronomical constants        
kb = 1.38065e-16 # boltzmann's constant
mh = 1.673558e-24 # mass of hydrogen atom in grams
sigma = 5.6704e-5 # stefan-Boltzmann constant

G=6.67408e-8 # Gravitational constant

earth_mass_to_g = 5.972e27
earth_radius_to_cm = 6.371e8
Mearth = earth_mass_to_g*1e-3 # kg
Rearth = earth_radius_to_cm*1e-2 # m
rho_earth_cgs = 5.51 # density of Earth
year_to_sec = 31557600. # number of seconds in a year
gyr_to_sec = 1e9*year_to_sec
day_to_sec = 86400. # number of secconds in a day

Tsun = 5778. #K
Rsun = 6.9551e8 # m
Msun = 1.9885e30 #kg
Lsun = 3.828e26 #W
Fxuv_earth = 0.67e-3 #W/m^2
Fxuv_earth_cgs = 4.666
au_to_cm = 1.496e13
AU = au_to_cm*1e-2
Period_Earth = 365.25 #days

#My units in SI
me = 5.972e24
re = 6.371e6

Ggrav=6.67408e-11
sigmasb=5.670374419e-8
gyr=gyr_to_sec
Rmol=8.31446261815324
Nav=6.02214076e23
Teq_e=5772.0*np.sqrt(1.362e9/(4.0*1.5e11))


# Standard energy-limited formalism
def massloss_energy_limited(
    fxuv   :float,
    mass   :float,
    radius :float,
    eff  :float = 1.0,
    beta :float = 1.0,
    ) -> float:
    """
    Returns the mass loss rate from a planet using the
    energy-limited model.
    Required parameters must be defined in the state,
    optional parameters can be provided through model keywords.

    Parameters
    ----------
    fxuv    : float, XUV flux of the star at planet orbit
    mass    : float, planet mass 
    radius  : float, planet radius 
    eff  : float|callable (optional),
            evaporation efficiency.
    beta    : float, Rxuv/Rpl
    Returns
    -------
    mloss  : float, mass loss rate in kg/s
    """
    mloss = (beta**2)*eff*np.pi*fxuv*(radius**3)/(Ggrav*mass)
    return mloss


#-- Owen & Wu 2017
def efficiency(mp,rp,mstar,sep):
    '''
    Efficiency of photoevaporative atmospheric H/He mass loss.
    Owen & Wu 2017
    '''
    v_esc = np.sqrt(2*Ggrav*mp / rp) #escape velocity at planet surface
    eff = 0.1 * (v_esc / 15e3)**(-2) #mass-loss efficiency, uncorrected
    
    #Roche lobe correction, Salz+ 2016a
    xi = (sep / rp) * (mp / (3*mstar))**(1/3)   #ξ
    k_diff = 1 - 3/(2 * xi) + 1/(2 * xi**3)     #fractional gravitational potential energy difference
    
    efficiency = eff/k_diff #corrected efficiency
    return efficiency

def k_diff(mp,rp,mstar,sep):
    '''
    fractional gravitational potential energy difference
    Salz+2016
    '''
        
    #Roche lobe correction, Salz+ 2016a
    xi = (sep / rp) * (mp / (3*mstar))**(1/3)   #ξ
    k_diff = 1 - 3/(2 * xi) + 1/(2 * xi**3)     #fractional gravitational potential energy difference
    
    return k_diff

def find_Lxuv(age,lstar):
    '''
    High-energy luminosity (X + UV) emitted by the star
    Sanz-Forçada+2011
    '''

    #L_xuv in non-saturation regime
    Lx = 1.89e21 * (age/Gyr)**(-1.55)
    Leuv = 10**(3.82) * Lx**0.86
    Lxuv = Lx + Leuv

    #L_xuv in saturation regime
    Lxsat = lstar * 6.3e-4
    Leuvsat = 10**(3.82) * Lxsat**0.86
    Lxuvsat = Lxsat + Leuvsat
    new_lxuv = np.minimum(Lxuv,Lxuvsat) #final Lxuv
    return new_lxuv

def mdot_ow17(fxuv,mstar,mp,rp,sep):
    '''
    Mass loss rate.
    Owen & Wu 2017
    '''
    eta = efficiency(mp,rp,mstar,sep)
    mdot = eta * np.pi * (fxuv*rp**3) / (Ggrav*mp)
    return mdot


# Differential escape of H and O
### following the formalism of Bolmont et al. 2017 (2017MNRAS.464.3728B)

# mass of H and O
amu = 1.66053906892e-26 # in kg
mH = amu*1.008
mO = amu*15.999

# stoechiometry of H2O molecule (molar)
XO = 1/3
XH = 2/3

kb = 1.380649e-23 #Boltzmann constant: J . K^-1
b_col = 2.2e21 #O on H collision parameter: m^-1 . sec^-1, from Kasting and Pollack 1983

def func_FH(df,b_col):
    '''
    Mass loss flux of hydrogen (kg/s/m^2).
    '''
    FM = df["Mdot"]/(4*np.pi*(df["Rp"]*re)**2)
    top = FM + mO*XO*(mO-mH) * b_col*df["grav"]/(kb*df["Teq"])
    bottom = mH + mO*XO/XH
    FH = top/bottom
    
    # df.loc[:,"FH"] = FH
    return FH

def func_mc(df,b_col):
    '''
    Cross-over mass.
        mc = +infty ---> rF = XO/XH = 0.5 ---> stoechiometric escape of H and O
        mc = mO     ---> rF = 0           ---> only H escape
        mc < mO     ---> rF = 0           ---> only H escape
    '''
    FH = func_FH(df,b_col)
    mc = mH + kb*df["Teq"]*FH / (b_col*df["grav"]*XH)

    # df.loc[:,"mc"] = mc
    # df.loc[:,"mc_amu"] = mc/amu
    return mc,mc/amu
    
def func_rF(df,b_col):
    '''
    Ratio of escape fluxes: rF = FO/FH
        rF = XO/XH = 0.5 ---> stoechiometric escape of H and O
        rF = 0           ---> only H escape
    '''
    mc, mc_amu = func_mc(df,b_col)
    rF = XO/XH * (mc - mO) / (mc - mH)
    rF = np.clip(rF,a_min = 0, a_max = None)

    # df.loc[:,"rF"] = rF
    return rF