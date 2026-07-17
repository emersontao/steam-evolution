# eXoPlanet MassLoss Module
#   VERSION:        SUMMER 2026
#   LAST UPDATED:   JUNE 17 2026, DESKTOP
#   INTERIOR:       Aguichine et al. 2025
#   UNITS:          MKS, also exports EARTH
#   Added Wind-AE!!!!
# //////////////////////////////////////// #

import os, sys
sys.path.insert(0, './Models')
import numpy as np
# import matplotlib.pyplot as plt
import swe
import windae_grid_interpolate as wgi

#constants and astronomical units
M_e   = 5.972e24  #Earth mass (kg)
R_e   = 6.371e6   #Earth radius (m)

M_sun = 1.988e30  #Solar mass (kg)
R_sun = 6.955e10  #Solar radius (m)
L_sun = 3.828e26  #luminosity of the Sun (Watts, kg⋅m^2⋅s^-3)
T_sun = 5778      #Solar temperature (K)


Fuv_e = 4.667     #W/m^2
au    = 1.496e11  #astronomical unit (m)
Gyr   = 3.153e16  #seconds in 1 billion years
G     = 6.674e-11 #gravitational constant (m^3⋅kg^-1⋅s^-2)
sigma_sb = 5.670374419e-8 #W⋅m−2⋅K−4 ; Stefan-Boltzmann constant

#Evolution for a single exoplanet (MKS)
#  Returns evolution file tracking age, mass, radius, wmf, mass-loss rate
def evolution(mp,rp,teq,wmf,sep,mstar,lstar):
    """
    mp: planet mass [kg]
    rp: planet radius [m]
    teq: equilibrium temperature [K]
    wmf: water-mass fraction
    sep: planet-star separation [m]
    mstar: stellar mass [kg]
    lstar: stellar luminosity [W]
    """
    #Input units: MKS

    #time values
    time_init = 1e-3 * Gyr      #1 Myr
    time_end = 5 * Gyr
    delta_time = 1e-4 * Gyr
    time_step_save = 1e-3 * Gyr #default: 1e-3 Gyr
    time_since_last_save = 0
    age = time_init #s

    #L*-dependent wrappers
    #https://en.wikipedia.org/wiki/Stellar_classification
    #0.43 comes from luminosity-relations
    mstar_cutoff = 0.43     #0.5585 -> midpoint of K-type min/max bolometric luminosities, solar masses. 
    if mstar/M_sun < mstar_cutoff:
        def radius(wmf,teq,mp,age):
            swerad = swe.radius_sweet(np.array(mp/M_e).flatten(), 
                                      np.array(0), np.array(0), 
                                      np.array(wmf), np.array(teq),
                                      np.array(age/Gyr))
            return swerad * R_e
    else:
        #Host star is G-TYPE
        def radius(wmf,teq,mp,age):
            swerad = swe.radius_sweet(np.array(mp/M_e).flatten(), 
                                      np.array(0), np.array(1), 
                                      np.array(wmf), np.array(teq),
                                      np.array(age/Gyr))
            return swerad * R_e
        
    mcore = mp * (1 - wmf)
    rsurf = swe.zeng_radius(mcore/M_e,0.325)
    grav = gravity(mp,rp)
    F_xuv = find_Fxuv(age,lstar,sep)
    # print(mcore)
    mdot = wgi.mdot(mstar/M_sun,teq,F_xuv/Fuv_e, grav, mp/M_e)
    err = wgi.error(mstar/M_sun,teq,F_xuv/Fuv_e, grav, mp/M_e)
    # mdot = wgi.mdot(mstar/M_sun,teq,F_xuv/Fuv_e,np.array([9]),np.array([2]))#,grav,mp/M_e)
    # err = wgi.error(mstar/M_sun,teq,F_xuv/Fuv_e,np.array([9]),np.array([2]))#,grav,mp/M_e)
    

    #create arrays
    time_saved = np.array([])
    mp_saved = np.array([])
    wmf_saved = np.array([])
    rp_saved = np.array([])
    mdot_saved = np.array([])
    error_saved = np.array([])

    #add the starting values
    time_saved = np.append(time_saved, age)
    mp_saved = np.append(mp_saved, mp)
    wmf_saved = np.append(wmf_saved, wmf)
    rp_saved = np.append(rp_saved, rp)
    mdot_saved = np.append(mdot_saved, mdot)
    error_saved = np.append(error_saved, err)
    iterations = 0

    # t_start = time.time()

    #simulation
    while age < time_end :
        #calculate timestep here
        mdot_old = mdot

        mdot = wgi.mdot(mstar/M_sun,teq,F_xuv/Fuv_e, grav, mp/M_e)
        err = wgi.error(mstar/M_sun,teq,F_xuv/Fuv_e, grav, mp/M_e)
        # print('mstar:', mstar/M_sun,'T_eq:',teq,'Fxuv:', F_xuv/Fuv_e, 'Mdot:', mdot, 'Errcode:', err)
        # break
        
        dtmin = (mp / mdot) * 1e-16
        if iterations > 0:
            np.seterr(divide='ignore') #ignore the divide by 0 errors
            dtmax = ( 2 * mp ) / np.abs(mdot - mdot_old)
            delta_time = 10**(0.5 * (np.log10(dtmin)) + 0.5*np.log10(dtmax))
            delta_time = min(delta_time,0.5*time_step_save)     #stops dt > time_step_save            
        else:
            delta_time = 10**(0.5 * (np.log10(dtmin)))

        if (delta_time + age) > time_end:
            #timestep failsafe
            delta_time = time_end - age


        #WMF Safety condition
        expected_wmf = 1 - (mcore/(mp - mdot*delta_time))
        if expected_wmf < 0.001:
            #Will fully deplete remaining 0.1% of water in ONE timestep.
            dt_new = (mp*wmf)/mdot
            # print(dt_new)
            # mdot_original = mdot

            age = age + dt_new
            mp = mp - mdot*dt_new 
            wmf = 1 - (mcore/mp) #which should be 0 now
            rp = rsurf

            # diditwork = 'Yes'
            # # print('wmf < 0 final rp:',rp)
            break
        
        #Update state
        age = age+delta_time
        mp = mp - mdot*delta_time
        wmf = 1 - (mcore/mp)
        rp = radius(wmf,teq,mp,age)

        grav = gravity(mp,rp)
        F_xuv = find_Fxuv(age,lstar,sep)
        time_since_last_save += delta_time
        iterations += 1

        #Break conditions
        if mp < 0 :
            break
        if rp < 0 :
            break
        if mdot < 0:
            break
        if mdot == 0:
            break
        #save to arrays every time step
        if time_since_last_save >= time_step_save :
            time_saved = np.append(time_saved, age)
            mp_saved = np.append(mp_saved, mp)
            rp_saved = np.append(rp_saved, rp)
            wmf_saved = np.append(wmf_saved, wmf)
            mdot_saved = np.append(mdot_saved, mdot)
            error_saved = np.append(error_saved, err)
            time_since_last_save = 0     #reset time since last save
        

    #add final points
    time_saved = np.append(time_saved, age)
    mp_saved = np.append(mp_saved, mp)
    wmf_saved = np.append(wmf_saved, wmf)
    rp_saved = np.append(rp_saved, rp)
    mdot_saved = np.append(mdot_saved, mdot)
    error_saved = np.append(error_saved, err)
    # print(reason)
    # end_t = time.time()
    #print(f'Runtime: {end_t - t_start:.2f}s')

    #save data
    # header = f'time [s]\tmass [kg]\twmf\tradius [m]\tmdot [kg]'
    # np.savetxt(f'./evolution-data/gp-evo/5000_xp/planet_1-1_evolution_data_wmf2.csv', \
    #         np.c_[time_saved, mp_saved, rp_saved, wmf_saved, mdot_saved], \
    #         fmt='%.9f', delimiter='\t', header=header)
    # print('File saved')

    return time_saved, mp_saved, rp_saved, wmf_saved, mdot_saved, error_saved

#Uniformly generated steam sub-Neptune planets for n planets
#  Makes file with: planet index, mass, radius, core mass, equilibrium temperature, water-mass fraction, scale height,
#  Jean's escape rate, distance from star, orbital period, star type (Mtype:0, Gtype:1), host star mass, host star luminosity
def planet_generator(n_planets, printtable=False):
    planetnum = np.arange(1,n_planets+1,1,dtype=int) #Assign a number to each random planet
    
    age_init = 1e-3 #Gyr, 1 Myr; consistent age
    
    # STELLAR PARAMETERS
    # Mstar Parameters drawn from Cifuentes et al. (2007) https://arxiv.org/pdf/2007.15077 (Table 6)
    mstar_gp = np.random.uniform(0.077,1.04,n_planets) * M_sun     # kg
    lstar_gp = np.array([])                             #W

    mp_gp = np.random.uniform(1.0,20.0,n_planets) * M_e     #M_e
    teq_gp = np.random.uniform(500.0,1500.0,n_planets)      #K, 750 initially
    wmf_gp = np.random.uniform(0.001,0.6,n_planets)
    rp_gp = np.array([])                                #m
    mcore_gp = np.array([])                             #M_e
    rsurf_gp = np.array([])                             #R_e
    period_gp = np.array([])                            #days
    fxuv_gp = np.array([])                              #W/m^2
    # core_r_gp = np.array([])
    # TEMPORARY reference: https://en.wikipedia.org/wiki/Stellar_classification, eventually use https://articles.adsabs.harvard.edu/pdf/1981A%26AS...46..193H
    mstar_cutoff = 0.43 #0.5585 -> midpoint of K-type min/max bolometric luminosities, solar masses. 0.43 comes from luminosity-relations
    for j in range(n_planets):
        # Luminosity for M and G
        # https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation

        mcore = mp_gp[j] * (1-wmf_gp[j])
        mcore_gp = np.append(mcore_gp,mcore)

        rsurf = swe.zeng_radius(mcore/M_e, 0.325)  #m, only accounts for rocky portion of planet
        rsurf_gp = np.append(rsurf_gp,rsurf)
        
        startype_gp = mstar_gp/M_sun > mstar_cutoff #actually a boolean mask
        if startype_gp[j]: 
            # Host star is M-TYPE
            lstar = 0.23 * L_sun * (mstar_gp[j] / M_sun)**2.3
            rp = swe.radius_m(wmf_gp[j],teq_gp[j],mp_gp[j]/M_e,age_init)*R_e #m
            T = period_solver_m(mstar_gp[j],teq_gp[j]) #days
        else:
            # Host star is G-TYPE
            lstar = L_sun * ((mstar_gp[j] / M_sun)**4)
            rp = swe.radius_g(wmf_gp[j],teq_gp[j],mp_gp[j]/M_e,age_init)*R_e #m
            T = period_solver_g(mstar_gp[j],teq_gp[j]) #days

        # append the star-dependent values
        lstar_gp = np.append(lstar_gp,lstar)
        rp_gp = np.append(rp_gp, rp)
        period_gp = np.append(period_gp,T)

    sep_gp = sep_from_teq(lstar_gp,teq_gp) #m, semi-major axis
    H_gp = scaleheight(teq_gp,rp_gp,mp_gp) #m, initial scale height of the atmosphere
    lambda_gp = rp_gp/H_gp #Jean's escape parameter
    grav_gp = gravity(mp_gp,rp_gp) #m/s^2, surface gravity
    fxuv_gp = find_Fxuv(1e-3*Gyr,lstar_gp,sep_gp)

    print(f'\n{n_planets} planets generated.')

    # //////////////////////////////////////// #
    # SAVE GENERATED DATA

    # Location details
    savepath = './evolution-data/paper/' #gp-evo/
    datafolder = f'{n_planets}_xp'
    location = savepath + datafolder
    
    # Create folder to save evolution data
    try:
        os.mkdir(location)
        print(f"Directory '{datafolder}' created successfully in {savepath}.")
    except FileExistsError:
        print(f"Directory '{datafolder}' already exists in {savepath}.")
    except Exception as e:
        print(f"An error occurred: {e}")

    #Exercise caution when modifying these!
    header = f'Planet/{n_planets}\tMp (kg)\tRp (m)\tCore M (kg)\tSurface Radius (m)\tTeq (K)\tWMF\tgravity(m/s^2)\tscale height (m)\tlambda\tsep (m)\tperiod (days)\tG-Star Type\tMstar (kg)\tLstar (W)\tFxuv (W/m^2)'
    header_e = f'Planet/{n_planets}\tMp (Me)\tRp (Re)\tCore M (Me)\tSurface Radius (Re)\tTeq (K)\tWMF\tgravity(m/s^2)\tscale height (m)\tlambda\tsep (au)\tperiod (days)\tG-Star Type\tMstar (Msun)\tLstar (Lsun)\tFxuv (W/m^2)'
    formats = ['%d','%.5e','%.5e','%.5e','%.5e','%.3f','%.5e','%.5e','%.3f','%.1f','%.5e','%.5f','%i','%.5e','%.5e','%.5e']
    formats_e = ['%d','%.5f','%.5f','%.5f','%.5f','%.3f','%.5f','%.5f','%.3f','%.1f','%.5f','%.3f','%i','%.5f','%.5f','%.5f']
    filename = f'{n_planets}_xp_gp_data.csv'
    filename_e = f'{n_planets}_xp_gp_data_unitEarth.csv'
    
    #in MKS units
    np.savetxt(f'{location}/{filename}', \
               np.c_[planetnum, mp_gp, rp_gp, mcore_gp, rsurf_gp, teq_gp, wmf_gp, grav_gp, H_gp, lambda_gp, \
                     sep_gp, period_gp, startype_gp, mstar_gp, lstar_gp, fxuv_gp], \
               fmt=formats, delimiter='\t', header=header) 

    #and in Earth/Sol units:    
    np.savetxt(f'{location}/{filename_e}', \
               np.c_[planetnum, mp_gp/M_e, rp_gp/R_e, mcore_gp/M_e, rsurf_gp/R_e, teq_gp, wmf_gp, grav_gp, H_gp, lambda_gp, \
                     sep_gp/au, period_gp, startype_gp, mstar_gp/M_sun, lstar_gp/L_sun, fxuv_gp], \
               fmt=formats_e, delimiter='\t', header=header_e)
    
    if printtable == True:
        # Display planet info (Earth units)

        print('\nPLANET POPULATION INFORMATION')
        from astropy.table import Table
        gp_table = Table()
        gp_table["Number"] = planetnum
        gp_table["Mass (M𛲜)"] = mp_gp/M_e
        gp_table["Radius (R𛲜)"] = rp_gp/R_e
        gp_table["Surface Radius (R𛲜)"] = rsurf_gp/R_e
        gp_table["Teq (K)"] = teq_gp
        gp_table["WMF"] = wmf_gp
        gp_table["Gravity (m/s^2)"] = grav_gp
        gp_table["Scale Height (km)"] = H_gp/1000
        gp_table["Lambda"] = lambda_gp
        gp_table["Period (days)"] = period_gp
        gp_table["Separation (au)"] = sep_gp/au
        gp_table["G-star Type"] = startype_gp
        gp_table["Lstar (L☉)"] = lstar_gp/L_sun
        gp_table["Mstar (M☉)"] = mstar_gp/M_sun
        gp_table["Fxuv (W/m^2)"] = fxuv_gp
        
        gp_table["Mass (M𛲜)"].format = "{:.3f}"
        gp_table["Radius (R𛲜)"].format = "{:.3f}"
        gp_table["Surface Radius (R𛲜)"].format = "{:.3f}"
        gp_table["Teq (K)"].format = "{:.3f}"
        gp_table["WMF"].format = "{:.3f}"
        gp_table["Gravity (m/s^2)"].format = "{:.3f}"
        gp_table["Scale Height (km)"].format = "{:.1f}"
        gp_table["Lambda"].format = "{:.3f}"
        gp_table["Period (days)"].format = "{:.1f}"
        gp_table["Separation (au)"].format = "{:.3f}"
        gp_table["Lstar (L☉)"].format = "{:.3f}"
        gp_table["Mstar (M☉)"].format = "{:.3f}"
        gp_table["Fxuv (W/m^2)"].format = "{:.3f}"
        print(gp_table)

    # print(len(lstar_gp))
    #Return units: MKS
    #          0        1      2        3         4        5       6       7       8       9        10       11          12         13         14    15
    return planetnum, mp_gp, rp_gp, mcore_gp, rsurf_gp, teq_gp, wmf_gp, grav_gp, H_gp, lambda_gp, sep_gp, period_gp, startype_gp, mstar_gp, lstar_gp, fxuv_gp





#calculate planet period
def period_solver_m(mstar,teq):
    Ls = L_sun*0.23*(mstar/M_sun)**2.3
    a = (Ls / (16 * np.pi * sigma_sb * teq**4))**(3/2)
    
    T = 365.25 * np.sqrt(M_sun * a / (mstar * au**3)) #days
    return T

def period_solver_g(mstar,teq):
    Ls = L_sun*(mstar/M_sun)**4
    a = (Ls/(16*np.pi*sigma_sb*teq**4))**(3/2)
    T = 365.25 * np.sqrt(M_sun * a / (mstar * au**3)) #days
    return T


#calculate planet gravity
def gravity(mp,rp):
    grav = (mp * G)/(rp**2)  #m/s
    return grav


#calculate scale height
def scaleheight(teq,rp,mp):
    mmolm = 18.01528 * 0.001 # kg/mol, mean molecular mass of the atmosphere (g/mol*0.001kg/g)
    R = 8.31446 #J/Kmol, molar gas constant
    g = gravity(mp,rp) # m/s, gravity of planet
    
    H = (teq*R)/(g*mmolm) # m
    return H


#calculate semi-major axis/equilibrium temperature
def sep_from_teq(lstar,teq):
    a = np.sqrt (lstar / (16*np.pi*sigma_sb * teq**4))
    return a

def teq_from_sep(lstar,sep):
    teq = (lstar / (16*np.pi*sigma_sb * sep**2))**0.25
    return teq


#find F_xuv at planet orbit
def find_Fxuv(age,lstar,sep):
    '''
    age:    system age [s]
    lstar:  stellar luminosity [W]
    sep:    semi-major axis [m]
    '''
    #SI units
    Fxuv =  find_Lxuv(age,lstar) / (4*np.pi*sep**2)

    return Fxuv


#-- Owen & Wu 2017
def efficiency(mp,rp,mstar,sep):
    #efficiency of photoevaporative atmospheric H/He mass loss
    #Owen & Wu 2017
    v_esc = np.sqrt(2*G*mp / rp) #escape velocity at planet surface
    eff = 0.1 * (v_esc / 15e3)**(-2) #mass-loss efficiency, uncorrected
    
    #Roche lobe correction, Salz+ 2016a
    xi = (sep / rp) * (mp / (3*mstar))**(1/3)   #ξ
    k_diff = 1 - 3/(2 * xi) + 1/(2 * xi**3)     #fractional gravitational potential energy difference

    efficiency = eff/k_diff #corrected efficiency
    return efficiency

def find_Lxuv(age,lstar):
    #high-energy luminosity (X + UV) emitted by the star
    #Sanz-Forçada+2011

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

def solve_mdot(mp,rp,mstar,sep,age,lstar):
    L_he = find_Lxuv(age,lstar)
    eta = efficiency(mp,rp,mstar,sep)
    mdot = eta * (L_he*rp**3) / (4*G*mp*sep**2)

    return mdot
