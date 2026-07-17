# Steam World Evolution interior model
#   VERSION:        SPRING 2026
#   LAST UPDATED:   MAY 12, 2026, DESKTOP
#   UNITS:          Earth
# https://github.com/an0wen/MARDIGRAS
# //////////////////////////////////////// #

import numpy as np 
from scipy.interpolate import RegularGridInterpolator
import sys
sys.path.insert(0, './Models')
#import iop_fint as iop

#physical constants
Ggrav = 6.67e-11 #gravitational constant m^3/kg/sec^2
Rmol = 8.314 #ideal gas constant (J/mol/K)

#astronomical constants
M_e = 5.97237e24 #Earth mass (kg)
R_e = 6.371e6 #Earth radius (m)
M_sun = 1.9885e30 #Sun mass (kg)
R_sun = 6.9551e10 #Sun radius (m)
L_sun = 3.828e26 #Total luminosity of the Sun (Watts = Joules/sec)
T_sun = 5778. #Temperature of the Sun (K)
au = 1.496e11 #Astronomical Unit (m)
Gyr = 3.15e16 #Seconds in 1 Gyr

#other constants
#L_he = 4.5e-6*L_sun + 4.6e-7*L_sun #EUV + X-Ray luminosities of the Sun (W)
eta = 0.1 #photoevaporation efficiency constant
sigmasb = 5.670374419e-8 # Stefan-Boltzmann constant  (W.m-2.K-4)




#-- SWE Aguichine et al. 2025
# Load the data
path_models = "./Models/"

path_sweet = path_models + "Aguichine2025_SWEET_all.dat"

# top of the atmosphere pressure (index)
sweet_top = np.array([0,1])
sweet_labels_top = ["20 mbar", "1 µbar"]
# host star type
sweet_labels_host = ["Type M", "Type G"]
# equilibrium temperature at zero albedo (K)
sweet_teqs = np.array([400, 500, 700, 900, 1100, 1300, 1500])
# bulk water mass fraction (%)
sweet_wmfs = np.array([0.1,1,10,20,30,40,50,60,70,80,90,100])
# total planet mass (Me)
sweet_masses = np.array([0.2       ,  0.254855  ,  0.32475535,  0.41382762,  0.52733018,
        0.67196366,  0.85626648,  1.09111896,  1.39038559,  1.77173358,
        2.25767578,  2.87689978,  3.66596142,  4.67144294,  5.95270288,
        7.58538038,  9.66586048, 12.31696422, 15.69519941, 20.        ])
# planet age (Gyr)
sweet_ages = np.array([0.001,0.0015,0.002,0.003,0.005,0.01,
                        0.02,0.03,0.05,
                        0.1,0.2,0.5,
                        1.0,2.0,5.0,
                        10,20])

# make dimension axes for interpolator
sweet_dim_wmf = sweet_wmfs.copy()/100
sweet_dim_teq = sweet_teqs.copy()
sweet_dim_mass = sweet_masses.copy()
sweet_dim_age = sweet_ages.copy()
sweet_dim_star = np.array([0,1])
sweet_dim_top = np.array([0,1])

# read radius tracks data and reshape
listrpfull1 = np.loadtxt(path_sweet,skiprows=36,unpack=True,usecols=(6))
listrpfull2 = np.loadtxt(path_sweet,skiprows=36,unpack=True,usecols=(7))
listrpfull = np.array((listrpfull1,listrpfull2))
listrpfull = np.delete(listrpfull, np.arange(17, listrpfull.size, 18))

sweet_data_radius = np.reshape(listrpfull,(2,2,12,7,20,17))
mask = np.isnan(sweet_data_radius)
sweet_data_radius[mask] = -1

# fill_value: value to return when extrapolating beyond grid validity range
fill_value = np.nan
interp_sweet = RegularGridInterpolator((sweet_dim_top,
                                      sweet_dim_star,
                                      sweet_dim_wmf,
                                      sweet_dim_teq,
                                      sweet_dim_mass,
                                      sweet_dim_age), 
                                     sweet_data_radius, method='slinear', bounds_error=False, fill_value=fill_value)


# The parametrized function to be plotted
def radius_sweet(x,top_sweet,star_sweet,wmf_sweet,teq_sweet,age_sweet):
    """
    x: mass (M_earth)

    top_sweet: top of the atmosphere pressure (index [0,1] for 20 milibar, 1 microbar)
    
    star_sweet: index [0,1] for M, G type host star
    
    wmf_sweet: water-mass fraction
    
    teq_sweet: equilibrium temperature (K)
    
    age_sweet: age of planet (Gyr)
    """
    input0 = np.stack((np.full(len(x),top_sweet),
                       np.full(len(x),star_sweet),
                       np.full(len(x),wmf_sweet),
                       np.full(len(x),teq_sweet),
                       x,
                       np.full(len(x),age_sweet)), axis=-1)
    return interp_sweet(input0)

#Host star is M-TYPE
def radius_m(wmf,teq,mp,age):
    swerad = radius_sweet(np.array(mp).flatten(), 
                                np.array(0), np.array(0), 
                                np.array(wmf), np.array(teq),
                                np.array(age))
    return swerad

#Host star is G-TYPE
def radius_g(wmf,teq,mp,age):
    swerad = radius_sweet(np.array(mp).flatten(), 
                                np.array(0), np.array(1), 
                                np.array(wmf), np.array(teq),
                                np.array(age))
    return swerad


#-- Rocky model Zeng 2016
#https://arxiv.org/pdf/1512.08827
path_zeng = path_models + "Zeng2016.dat"

#open files
list_zeng_fe_m,list_zeng_fe_r = np.loadtxt(path_models+"zeng2016-iron.dat",unpack=True,usecols=(0,1))
list_zeng_ea_m,list_zeng_ea_r = np.loadtxt(path_models+"zeng2016-earth.dat",unpack=True,usecols=(0,1))
list_zeng_mg_m,list_zeng_mg_r = np.loadtxt(path_models+"zeng2016-rock.dat",unpack=True,usecols=(0,1))

list_zeng_masses = np.logspace(np.log10(0.01), np.log10(100.0), 30)

#reshape radius data to the same vector
list_zeng_fe_radii = np.interp(list_zeng_masses, list_zeng_fe_m, list_zeng_fe_r)
list_zeng_ea_radii = np.interp(list_zeng_masses, list_zeng_ea_m, list_zeng_ea_r)
list_zeng_mg_radii = np.interp(list_zeng_masses, list_zeng_mg_m, list_zeng_mg_r)

#create interpolator
dimcmf_zeng = np.array([0.0,0.325,1.0])
data_zeng = np.vstack((list_zeng_mg_radii, list_zeng_ea_radii,list_zeng_fe_radii)).T
interp_zeng = RegularGridInterpolator((list_zeng_masses,dimcmf_zeng), data_zeng, method='linear', bounds_error=False, fill_value=None)

def zeng_radius(mp,cmf):
    return interp_zeng((mp,cmf)).item(0) * R_e #m





#-- WMF SOLVER
# invert the radius_iop equation to solve for wmf
# takes in mkgs units

#dichotomy/bisection root-finding method
def bisection(func,a,b,precision, max_steps):
    steps = 0
    for _ in range(max_steps):
        c = (a + b) / 2
        steps += 1

        if abs(a - b) < precision:
            #print(f'[BISECTION: SUCCESS]\nSteps taken = {steps} \nInterval size = { 1/(2**steps) }')
            return c
        if abs(func(c)) < precision:
            return c
        if func(c) < 0:
            a = c
        else:
            b = c
    
    #print('ERROR: Did not converge?')
    #print(f'[BISECTION: FAILED]\nDid not converge\nSteps taken = {steps} \nInterval size = { 1/(2**steps) }')
    return c



def solve_wmf_m(rp,teq,mp,age,precision=1e-8,max_steps=100):
    #M-type stars
    a, b = 0, 1 #define interval
    
    rpmax = radius_sweet(mp, 0, 0, 1, teq, age)
    rpmin = radius_sweet(mp, 0, 0, 0.001, teq, age)

    if rp > rpmax:
        rp = rpmax
        print('Radius > 100% Water')
        return 1
    
    if rp < rpmin:
        rp = rpmin
        print('Radius < 0.1% Water')
        return 0

    def objective(wmf):
        return radius_sweet(mp, 0, 0, wmf, teq, age) - rp
    
    #make sure the sign changes between endpoints
    wmf_0 = objective(0)
    wmf_1 = objective(1)
    #print(f'[SOLVE WMF]  WMF(0.0) = {wmf_0}\tWMF(1.0) = {wmf_1}')
    if wmf_0 * wmf_1 > 0:
        raise ValueError('The function does not change sign in the interval [',a,',',b,']')
    
    wmf = bisection(objective,0,1,precision,max_steps)
    
    # #IF USING EXISTING EXOPLANET: )
    # # check for radius
    # calculated_rp = interp_radius_m([wmf,teq,mp,age])
    # if rp > calculated_rp :
    #     rp = calculated_rp
    #     print('WARNING: Calculated radius override given radius')
    #     wmf = bisection(objective,0,1,precision,max_steps)      #recalculate wmf

    #print(f'WMF: {wmf:.4f}')
    return wmf

def solve_wmf_g(rp,teq,mp,age,precision=1e-8,max_steps=100):
    # G-type stars
    a, b = 0, 1 #define interval
        
    rpmax = radius_sweet(mp,0,1,1,teq,age)
    rpmin = radius_sweet(mp,0,1,0.001,teq,age)

    if rp > rpmax:
        rp = rpmax
        print('Radius > 100% Water')
        return 1
    
    if rp < rpmin:
        rp = rpmin
        print('Radius < 0% Water')
        return 0

    def objective(wmf):
        return radius_sweet(wmf,teq,mp,age) - rp
    
    #make sure the sign changes between endpoints
    wmf_0 = objective(0)
    wmf_1 = objective(1)
    #print(f'[SOLVE WMF]  WMF(0.0) = {wmf_0}\tWMF(1.0) = {wmf_1}')
    if wmf_0 * wmf_1 > 0:
        raise ValueError('The function does not change sign in the interval [0, 1]')
    
    wmf = bisection(objective,0,1,precision,max_steps)
    
    # #IF USING EXISTING EXOPLANET:
    # # check for radius
    # calculated_rp = interp_radius_m([wmf,teq,mp,age])
    # if rp > calculated_rp :
    #     rp = calculated_rp
    #     print('WARNING: Calculated radius override given radius')
    #     wmf = bisection(objective,0,1,precision,max_steps)      #recalculate wmf

    #print(f'WMF: {wmf:.4f}')
    return wmf
