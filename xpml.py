# eXoPlanet MassLoss Module
#   VERSION:        SPRING 2025
#   LAST UPDATED:   JULY 25 2025, DESKTOP
#   INTERIOR:       Aguichine et al. 2025
#   UNITS:          MKS, also exports EARTH
#   Workshopping steam-related names
# //////////////////////////////////////// #

import os, sys
sys.path.insert(0, './Models')
import numpy as np
# import matplotlib.pyplot as plt
import swe

#constants and astronomical units
M_e   = 5.972e24  #Earth mass (kg)
R_e   = 6.371e6   #Earth radius (m)

M_sun = 1.988e30  #Solar mass (kg)
R_sun = 6.955e10  #Solar radius (m)
L_sun = 3.828e26  #luminosity of the Sun (Watts, kg⋅m^2⋅s^-3)
T_sun = 5778      #Solar temperature (K)

au    = 1.496e11  #astronomical unit (m)
Gyr   = 3.153e16  #seconds in 1 billion years
G     = 6.674e-11 #gravitational constant (m^3⋅kg^-1⋅s^-2)
sigma_sb = 5.670374419e-8 #W⋅m−2⋅K−4 ; Stefan-Boltzmann constant

#Evolution for a single exoplanet (MKS)
#  Returns evolution file tracking age, mass, radius, wmf, mass-loss rate
def evolution(mp,rp,teq,wmf,sep,mstar,lstar):
    #Input units: MKS

    #time values
    time_init = 1e-2 * Gyr      #10 Myr
    time_end = 20 * Gyr
    delta_time = 1e-4 * Gyr
    time_step_save = 1e-3 * Gyr #default: 1e-3 Gyr
    time_since_last_save = 0
    age = time_init #s

    #L*-dependent wrappers
    #https://en.wikipedia.org/wiki/Stellar_classification
    #0.43 comes from luminosity-relations
    mstar_cutoff = 0.43     #0.5585 -> midpoint of K-type min/max bolometric luminosities, solar masses. 
    if mstar/M_sun < mstar_cutoff:
        #Host star is M-TYPE
        def swe_radius(wmf,teq,mp,age):
            return swe.radius_m(wmf,teq,(mp/M_e),(age/Gyr)) * R_e
    else:
        #Host star is G-TYPE
        def swe_radius(wmf,teq,mp,age):
            return swe.radius_g(wmf,teq,(mp/M_e),(age/Gyr)) * R_e
        
    mcore = mp * (1 - wmf)
    rsurf = swe.zeng_radius(mcore/M_e,0.325)
    # print(mcore)
    mdot = solve_mdot(mp,rp,mstar,sep,age,lstar)
    # rsurf = swe.zeng_radius(mcore/M_e, 0.325) *R_e  #m, only if rsurf isn't already being tracked


    #create arrays
    time_saved = np.array([])
    mp_saved = np.array([])
    wmf_saved = np.array([])
    rp_saved = np.array([])
    mdot_saved = np.array([])

    #add the starting values
    time_saved = np.append(time_saved, age)
    mp_saved = np.append(mp_saved, mp)
    wmf_saved = np.append(wmf_saved, wmf)
    rp_saved = np.append(rp_saved, rp)
    mdot_saved = np.append(mdot_saved, mdot)
    iterations = 0

    # t_start = time.time()

    #simulation
    while age < time_end :
        #calculate timestep here
        mdot_old = mdot
        mdot = solve_mdot(mp,rp,mstar,sep,age,lstar)

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
            # reason = 'wmf < 0.001'

            break
        

        #Update state
        age = age+delta_time
        mp = mp - mdot*delta_time
        wmf = 1 - (mcore/mp)
        rp = swe_radius(wmf,teq,mp,age)
        time_since_last_save += delta_time
        iterations += 1
        

        #Break conditions
        if mp < 0 :
            # reason = 'mp < 0'
            break
        if rp < 0 :
            # reason = 'rp < 0'
            break
        if mdot < 0:
            # reason = 'mdot < 0'
            break
        if mdot == 0:
            # reason = 'mdot = 0'
            break
        # if wmf < 0 :
        #     reason = 'wmf < 0'
        #     # print('wmf break')
        #     break


        #save to arrays every time step
        if time_since_last_save >= time_step_save :
            time_saved = np.append(time_saved, age)
            mp_saved = np.append(mp_saved, mp)
            rp_saved = np.append(rp_saved, rp)
            wmf_saved = np.append(wmf_saved, wmf)
            mdot_saved = np.append(mdot_saved, mdot)
            time_since_last_save = 0     #reset time since last save
        
        # #debugging stuff
        # diditwork = 'No'
        # reason = 'Age'
        # dt_new = np.nan

    #add final points
    time_saved = np.append(time_saved, age)
    mp_saved = np.append(mp_saved, mp)
    wmf_saved = np.append(wmf_saved, wmf)
    rp_saved = np.append(rp_saved, rp)
    mdot_saved = np.append(mdot_saved, mdot)
    
    # end_t = time.time()
    #print(f'Runtime: {end_t - t_start:.2f}s')

    #save data
    # header = f'time [s]\tmass [kg]\twmf\tradius [m]\tmdot [kg]'
    # np.savetxt(f'./evolution-data/gp-evo/5000_xp/planet_1-1_evolution_data_wmf2.csv', \
    #         np.c_[time_saved, mp_saved, rp_saved, wmf_saved, mdot_saved], \
    #         fmt='%.9f', delimiter='\t', header=header)
    # print('File saved')

    return time_saved, mp_saved, rp_saved, wmf_saved, mdot_saved


#Uniformly generated steam sub-Neptune planets for n planets
#  Makes file with: planet index, mass, radius, core mass, equilibrium temperature, water-mass fraction, scale height,
#  Jean's escape rate, distance from star, orbital period, star type (Mtype:0, Gtype:1), host star mass, host star luminosity
def planet_generator(n_planets):
    planetnum = np.arange(1,n_planets+1,1,dtype=int) #Assign a number to each random planet
    
    age_init = 1e-2 #Gyr, 10 Myr; consistent age
    
    # STELLAR PARAMETERS
    # Mstar Parameters drawn from Cifuentes et al. (2007) https://arxiv.org/pdf/2007.15077 (Table 6)
    mstar_gp = np.random.uniform(0.077,1.040,n_planets) * M_sun     # kg
    lstar_gp = np.array([])                             #W

    mp_gp = np.random.uniform(1,12,n_planets) * M_e     #M_e
    teq_gp = np.random.uniform(750,1200,n_planets)      #K, 750 initially
    wmf_gp = np.random.uniform(0.001,0.6,n_planets)
    rp_gp = np.array([])                                #m
    mcore_gp = np.array([])                             #M_e
    rsurf_gp = np.array([])                             #R_e
    period_gp = np.array([])                            #days
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
        
        startype_gp = mstar_gp/M_sun > 0.43 #actually a boolean mask
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
    grav_gp = grav(mp_gp,rp_gp) #m/s^2, surface gravity

    print(f'\n{n_planets} planets generated.')

    # //////////////////////////////////////// #
    # SAVE GENERATED DATA

    # Location details
    savepath = './evolution-data/gp-evo/'
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
    header = f'Planet/{n_planets}\tMp (kg)\tRp (m)\tCore M (kg)\tSurface Radius (m)\tTeq (K)\tWMF\tgravity(m/s^2)\tscale height (m)\tlambda\tsep (m)\tperiod (days)\tG-Star Type\tMstar (kg)\tLstar (W)'
    header_e = f'Planet/{n_planets}\tMp (Me)\tRp (Re)\tCore M (Me)\tSurface Radius (Re)\tTeq (K)\tWMF\tgravity(m/s^2)\tscale height (m)\tlambda\tsep (au)\tperiod (days)\tG-Star Type\tMstar (Msun)\tLstar (Lsun)'
    formats = ['%d','%.5e','%.5e','%.5e','%.5e','%.3f','%.5e','%.5e','%.3f','%.1f','%.5e','%.5f','%i','%.5e','%.5e']
    formats_e = ['%d','%.5f','%.5f','%.5f','%.5f','%.3f','%.5f','%.5f','%.3f','%.1f','%.5f','%.3f','%i','%.5f','%.5f']
    filename = f'{n_planets}_xp_gp_data.csv'
    filename_e = f'{n_planets}_xp_gp_data_unitEarth.csv'
    
    #in MKS units
    np.savetxt(f'{location}/{filename}', \
               np.c_[planetnum, mp_gp, rp_gp, mcore_gp, rsurf_gp, teq_gp, wmf_gp, grav_gp, H_gp, lambda_gp, \
                     sep_gp, period_gp, startype_gp, mstar_gp, lstar_gp], \
               fmt=formats, delimiter='\t', header=header) 

    #and in Earth/Sol units:    
    np.savetxt(f'{location}/{filename_e}', \
               np.c_[planetnum, mp_gp/M_e, rp_gp/R_e, mcore_gp/M_e, rsurf_gp/R_e, teq_gp, wmf_gp, grav_gp, H_gp, lambda_gp, \
                     sep_gp/au, period_gp, startype_gp, mstar_gp/M_sun, lstar_gp/L_sun], \
               fmt=formats_e, delimiter='\t', header=header_e)
    
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
    print(gp_table)

    # print(len(lstar_gp))
    #Return units: MKS
    #          0        1      2        3         4        5       6       7       8       9        10       11          12         13         14
    return planetnum, mp_gp, rp_gp, mcore_gp, rsurf_gp, teq_gp, wmf_gp, grav_gp, H_gp, lambda_gp, sep_gp, period_gp, startype_gp, mstar_gp, lstar_gp


#Reformatted NEA data files for xp_evolution
#  Takes in values in astronomical units. only needs not null values for:
#  (3), period, semi-major axis, rp R_e, mp M_e, (2), teq, (4), mstar, lstar log
def nea_reformatter(filename):
    #from NEA:
    loadin_path = './evolution-data/real-evo/nea/'+filename
    period_gp,sep_gp,rp_gp,mp_gp,teq_gp,mstar_gp,lstar_gp = np.loadtxt(loadin_path,dtype='float',delimiter=',',skiprows=(41),usecols=(4,5,6,7,10,15,16),unpack=True)
    # period_gp,sep_gp,rp_gp,mp_gp,teq_gp,mstar_gp,lstar_gp = np.loadtxt(loadin_path,skiprows=(39),usecols=(4,5,6,7,10,15,16),unpack=True)
    n_planets = len(period_gp)
    planetnum = np.arange(1,n_planets+1,1,dtype='int')
    mp_gp = mp_gp * M_e
    rp_gp = rp_gp * R_e
    teq_gp = teq_gp

    mstar_gp = mstar_gp * M_sun
    lstar_gp = L_sun * 10**(lstar_gp)
    sep_gp = sep_gp * au

    wmf_gp = np.array([])
    rsurf_gp = np.array([])
    mcore_gp = np.array([])
    age_init = 1e-2 * Gyr

    startype_gp = mstar_gp/M_sun > 0.43 #actually a mask
    # startype_gp[mask_gstar] = 1

    for i in range(n_planets):
        if startype_gp[i]:
            # Host star is G-TYPE
            # startype_gp[i] = 1
            wmf = swe.solve_wmf_g(rp_gp[i],teq_gp[i],mp_gp[i],age_init)
        else:
            # Host star is M-TYPE
            wmf = swe.solve_wmf_m(rp_gp[i],teq_gp[i],mp_gp[i],age_init)
        
        rsurf = swe.zeng_radius(mp_gp/M_e,0.325) * R_e
        mcore = mp_gp[i] * (1-wmf)

        rsurf_gp = np.append(rsurf_gp, rsurf)
        mcore_gp = np.append(mcore_gp, mcore)
        wmf_gp = np.append(wmf_gp, wmf)
    
    grav_gp = grav(mp_gp,rp_gp)
    H_gp = scaleheight(teq_gp,rp_gp,mp_gp)
    lambda_gp = H_gp/rp_gp
    
    # //////////////////////////////////////// #
    # SAVE MODIFIED

    # Location details
    savepath = './evolution-data/real-evo/nea/'
    identifier = '071625'
    datafolder = f'{n_planets}_xp_{identifier}'
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
    header = f'Planet/{n_planets}\tMp (kg)\tRp (m)\tCore M (kg)\tSurface Radius (m)\tTeq (K)\tWMF\tgravity(m/s^2)\tscale height (m)\tlambda\tsep (m)\tperiod (days)\tStar Type\tMstar (kg)\tLstar (W)'
    header_e = f'Planet/{n_planets}\tMp (Me)\tRp (Re)\tCore M (Me)\tSurface Radius (Re)\tTeq (K)\tWMF\tgravity(m/s^2)\tscale height (m)\tlambda\tsep (au)\tperiod (days)\tStar Type\tMstar (Msun)\tLstar (Lsun)'
    formats = ['%d','%.5e','%.5e','%.5e','%.5e','%.3f','%.5e','%.5e','%.3f','%.1f','%.5e','%.5f','%i','%.5e','%.5e']
    formats_e = ['%d','%.5f','%.5f','%.5f','%.5f','%.3f','%.5f','%.5f','%.3f','%.1f','%.5f','%.3f','%i','%.5f','%.5f']
    filename_r = f'{n_planets}_xp_{filename}.csv'
    filename_e = f'{n_planets}_xp_{filename}_unitEarth.csv'
    
    #in MKS units
    np.savetxt(f'{location}/{filename_r}', \
               np.c_[planetnum, mp_gp, rp_gp, mcore_gp, rsurf_gp, teq_gp, wmf_gp, grav_gp, H_gp, lambda_gp, \
                     sep_gp, period_gp, startype_gp, mstar_gp, lstar_gp], \
               fmt=formats, delimiter='\t', header=header) 

    #and in Earth/Sol units:    
    np.savetxt(f'{location}/{filename_e}', \
               np.c_[planetnum, mp_gp/M_e, rp_gp/R_e, mcore_gp/M_e, rsurf_gp/R_e, teq_gp, wmf_gp, grav_gp, H_gp, lambda_gp, \
                     sep_gp/au, period_gp, startype_gp, mstar_gp/M_sun, lstar_gp/L_sun], \
               fmt=formats_e, delimiter='\t', header=header_e)


    #Return units: MKS
    #          0        1      2        3         4        5       6       7        8       9        10       11          12         13         14
    return planetnum, mp_gp, rp_gp, mcore_gp, rsurf_gp, teq_gp, wmf_gp, grav_gp, H_gp, lambda_gp, sep_gp, period_gp, startype_gp, mstar_gp, lstar_gp





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
def grav(mp,rp):
    grav = (mp * G)/(rp**2)  #m/s
    return grav

#calculate scale height
def scaleheight(teq,rp,mp):
    mmolm = 18.01528 * 0.001 # kg/mol, mean molecular mass of the atmosphere (g/mol*0.001kg/g)
    R = 8.31446 #J/Kmol, molar gas constant
    g = grav(mp,rp) # m/s, gravity of planet
    
    H = (teq*R)/(g*mmolm) # m

    return H

#calculate semi-major axis/equilibrium temperature
def sep_from_teq(lstar,teq):
    a = np.sqrt (lstar / (16*np.pi*sigma_sb * teq**4))
    return a

def teq_from_sep(lstar,sep):
    teq = (lstar / (16*np.pi*sigma_sb * sep**2))**0.25
    return teq


#calculate surface radius (NOT reliable)
def surface_radius(H, wmf, rp):
    #calculates the radius of the core/rocky surface of a steam world

    #atmosphere radius
    #note that this is depth, not really radius. Can also "read the actual-
    # "-R_atm between 1000 bar and 0.1 Pa from the output file" -Artem  (?)
    P_rcb = 1000        #1e8 Pa,    pressure at radiative convective boundary (rcb), from optically thin to thick
    P_transit = 0.001    # 0.1 #Pa, 1 μbar
    r_atm = H * np.log(P_rcb/P_transit) #m, radius between 1000 bar and 1 μbar

    #envelope radius, VERY rough estimate
    r_env = wmf * rp #assuming wmf is maintained in the interior envelope and is proportional to radius

    #rocky core
    r_0 = rp - r_env - r_atm #m; hydrosphere: envelope(interior) + atmosphere

    return r_0


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

def L_xuv(age,lstar):
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
    L_he = L_xuv(age,lstar)
    eta = efficiency(mp,rp,mstar,sep)
    mdot = eta * (L_he*rp**3) / (4*G*mp*sep**2)

    return mdot
