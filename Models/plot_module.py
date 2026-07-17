#last updated: JUNE 17, 2026 on BOTH

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os

#suggested run pattern
def run(n_planets=5000, datapath="./evolution-data/paper/", savepath="./plots/paper/",
                      trails=True, planets_plotted=200, 
                      kde=True, escapetype="H2O", catalog="NEA"):
    """
    n_planets: total planets from that simulation. default is 5000
    datapath: where the simulations are located
    savepath: where plots are output
    trails: do you want a trails plot?
    planets_plotted: number of planets to visualize [trails]
    
    kde: do you want a kde plot?
    escapetype: H-He or H2O, default is H2O [kde]
    catalog: NEA or PlanetS, default is NEA for now [kde]

    Default is to plot trails for H-He, H2O and KDE for H2O.
    """
    #constants and astronomical units
    M_e   = 5.972e24  #Earth mass (kg)
    R_e   = 6.371e6   #Earth radius (m)

    #MARDIGRAS stuff
    xmin, xmax, nx = 0.1, 19.99999, 50 # 0.5, 20
    ymin, ymax  = .75, 4.25

    path_models='./Models/'
    x = np.logspace(np.log10(xmin), np.log10(xmax), nx)

    #MARDIGRAS stuff
    xmin, xmax, nx = 0.1, 19.99999, 50 # 0.5, 20
    ymin, ymax  = .75, 4.25

    ##############################################
    #
    #   MAKE SWEET INTERPOLATOR (Aguichine+2025)

    path_sweet = path_models + "Aguichine2025_SWEET_all.dat"

    # top of the atmosphere pressure (index)
    # equilibrium temperature at zero albedo (K)
    sweet_teqs = np.array([400, 500, 700, 900, 1100, 1300, 1500])
    # bulk water mass fraction (%)
    sweet_wmfs = np.array([0.1,1,10,20,30,40,50,60,70,80,90,100])
    # total planet mass (Me)
    sweet_masses = np.array([0.2       ,  0.254855  ,  0.32475535,  0.41382762,  0.52733018,
            0.67196366,  0.85626648,  1.09111896,  1.39038559,  1.77173358,
            2.25767578,  2.87689978,  3.66596142,  4.67144294,  5.95270288,
            7.58538038,  9.66586048, 12.31696422, 15.69519941, 20.        ])
    sweet_ages = np.array([0.001,0.0015,0.002,0.003,0.005,0.01,
                            0.02,0.03,0.05,
                            0.1,0.2,0.5,
                            1.0,2.0,5.0,
                            10,20]) #Gyr

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
    def radius_m(x,wmf_swe,teq_swe,age_swe):
        input0 = np.stack([x, np.full(len(x),0), np.full(len(x),0), np.full(len(x),wmf_swe), np.full(len(x),teq_swe), np.full(len(x),age_swe)],axis=-1)
        # input0 = np.stack(0, 0, (np.full(len(x),wmf_swe),np.full(len(x),teq_swe),x,np.full(len(x),age_swe)), axis=-1)
        return interp_sweet(input0)

    def radius_g(x,wmf_swe,teq_swe,age_swe):
        input0 = np.stack([x, np.full(len(x),0), np.full(len(x),1), np.full(len(x),wmf_swe), np.full(len(x),teq_swe), np.full(len(x),age_swe)],axis=-1)
        # input0 = np.stack(0, 1, (np.full(len(x),wmf_swe),np.full(len(x),teq_swe),x,np.full(len(x),age_swe)), axis=-1)
        return interp_sweet(input0)


    # Define initial parameters
    init_host_swe = 0
    init_wmf_swe = 0.5
    init_teq_swe = 600.0
    init_age_swe = 1.0

    init_wmf_swe100 = 1.0

    ##############################################
    # Load Zeng data
    # path_zeng = path_models + "Zeng2016.dat"

    # # Load curves from Zeng et al. 2016
    # zeng_mass,zeng_purefe,zeng_rock,zeng_50wat,zeng_100wat,zeng_earth = np.loadtxt(path_zeng,delimiter="\t",skiprows=1,unpack=True)

    # open files
    list_zeng_fe_m,list_zeng_fe_r = np.loadtxt(path_models+"zeng2016-iron.dat",unpack=True,usecols=(0,1))
    list_zeng_ea_m,list_zeng_ea_r = np.loadtxt(path_models+"zeng2016-earth.dat",unpack=True,usecols=(0,1))
    list_zeng_mg_m,list_zeng_mg_r = np.loadtxt(path_models+"zeng2016-rock.dat",unpack=True,usecols=(0,1))

    list_zeng_masses = np.logspace(np.log10(0.01), np.log10(100.0), 30)

    # reshape radius data to the same vector
    list_zeng_fe_radii = np.interp(list_zeng_masses, list_zeng_fe_m, list_zeng_fe_r)
    list_zeng_ea_radii = np.interp(list_zeng_masses, list_zeng_ea_m, list_zeng_ea_r)
    list_zeng_mg_radii = np.interp(list_zeng_masses, list_zeng_mg_m, list_zeng_mg_r)

    # create interpolator
    dimcmf_zeng = np.array([0.0,0.325,1.0])
    data_zeng = np.vstack((list_zeng_mg_radii, list_zeng_ea_radii,list_zeng_fe_radii)).T
    interp_zeng = RegularGridInterpolator((list_zeng_masses,dimcmf_zeng), data_zeng, method='linear', bounds_error=False, fill_value=None)

    # parameterized func to plot
    def rad_zeng(x,cmf):
        input0 = np.stack((x,np.full(len(x),cmf)), axis=-1)
        rp = interp_zeng(input0)
        return rp

    init_cmf_zeng = 0.325

    # Zeng fixed
    zeng_width = 1.0

    if not trails and not kde:
        print("what is wrong with you")

        exit()

    plt.clf()
    #create datasets for BOTH H2O and H-He
    #masses
    init_mp = np.array([])
    H2O_fin_mp = np.array([])
    HHe_fin_mp = np.array([])

    #radii
    init_rp = np.array([])
    H2O_fin_rp = np.array([])
    HHe_fin_rp = np.array([])

    identifier = f'{n_planets}_xp'

    population_file = datapath + identifier + '/5000_xp_gp_data.csv'
    planetnum, mp_gp, rp_gp, mcore_gp, rsurf_gp, teq_gp, wmf_gp, grav_gp, H_gp, lambda_gp, sma_gp, period_gp, startype_gp, mstar_gp, lstar_gp, fxuv_gp = np.loadtxt(population_file,skiprows=1,unpack=True)


    if trails:
        plt.clf()
        # start plot
        fig, ax = plt.subplots(2,1, figsize=(7,10))#, sharex=True)
        fig.subplots_adjust(hspace=0) 

        # loop to import each planet's information
        # arrays are in SI for consistency
        for i in range (planets_plotted-1):
            planetnum = i+1
            
            #add initial values
            init_mp = np.append(init_mp, mp_gp[planetnum])
            init_rp = np.append(init_rp, rp_gp[planetnum])

            #H2O final values
            H2O_data = f'{datapath}/{identifier}/H2O/planet_{planetnum}-{n_planets}_evolution_data.csv' #imported in Earth units
            H2O_mp_arr, H2O_rp_arr = np.loadtxt(H2O_data,usecols=(1,2),unpack=True) #,skiprows=1

            H2O_fin_mp = np.append(H2O_fin_mp, H2O_mp_arr[-1])
            H2O_fin_rp = np.append(H2O_fin_rp, H2O_rp_arr[-1])
            
            #hydrogen-helium final values
            HHe_data = f'{datapath}/{identifier}/H-He/planet_{planetnum}-{n_planets}_evolution_data.csv' #imported in Earth units
            HHe_mp_arr, HHe_rp_arr = np.loadtxt(HHe_data,usecols=(1,2),unpack=True) #,skiprows=1

            HHe_fin_mp = np.append(HHe_fin_mp, HHe_mp_arr[-1]) #[-2] to evade nan issues
            HHe_fin_rp = np.append(HHe_fin_rp, HHe_rp_arr[-1])
            
            #plot
            ax[0].plot(H2O_mp_arr/M_e, H2O_rp_arr/R_e, c="#FFB451", linewidth=1)
            ax[1].plot(HHe_mp_arr/M_e, HHe_rp_arr/R_e, c='#FFB451', linewidth=1)

        #last point for the legend
        init_mp = np.append(init_mp, mp_gp[planets_plotted])
        init_rp = np.append(init_rp, rp_gp[planets_plotted])

        #H2O final values
        H2O_data = f'{datapath}/{identifier}/H2O/planet_{planets_plotted}-{n_planets}_evolution_data.csv' #imported in Earth units
        H2O_mp_arr, H2O_rp_arr = np.loadtxt(H2O_data,usecols=(1,2),unpack=True) #,skiprows=1

        H2O_fin_mp = np.append(H2O_fin_mp, H2O_mp_arr[-1])
        H2O_fin_rp = np.append(H2O_fin_rp, H2O_rp_arr[-1])

        #hydrogen-helium final values
        HHe_data = f'{datapath}/{identifier}/H-He/planet_{planets_plotted+200}-{n_planets}_evolution_data.csv' #imported in Earth units
        HHe_mp_arr, HHe_rp_arr = np.loadtxt(HHe_data,usecols=(1,2),unpack=True) #,skiprows=1

        HHe_fin_mp = np.append(HHe_fin_mp, HHe_mp_arr[-1]) #[-2] to evade nan issues
        HHe_fin_rp = np.append(HHe_fin_rp, HHe_rp_arr[-1])


        # ------------------------------------------------ #
        #plot
        ax[0].plot(H2O_mp_arr/M_e, H2O_rp_arr/R_e, c="#FFB451", linewidth=1, label="Simulated evolution")
        ax[1].plot(HHe_mp_arr/M_e, HHe_rp_arr, c="#FFB451", linewidth=1,label="Simulated evolution")

        # -------------------------------
        #finish the plot
        ymin = min(H2O_fin_rp)/R_e - 0.25
        ymax = max(H2O_fin_rp)/R_e + 0.5
        # print(max(H2O_fin_rp/R_e),max(H2O_fin_rp)/R_e)
        ax[0].set_title(f"Simulated Evolution: {planets_plotted} Steam Worlds")
        ax[0].set(xlim=(0.25,20.25), xticklabels=[], ylim=(ymin,ymax), ylabel=r'Radius [$R_\oplus$]')
        ax[0].scatter(init_mp/M_e, init_rp/R_e, c='#D13242', alpha=1, s=15, edgecolor='none', zorder=50, label='Initial state')
        ax[0].scatter(H2O_fin_mp/M_e, H2O_fin_rp/R_e, c='#76AA27', alpha=1, s=10, edgecolor='none', zorder=30, label='Final state')
        ax[0].plot(list_zeng_masses,list_zeng_ea_radii,linewidth=zeng_width,color='brown',alpha=0.75,zorder=-100)#,label='Earth-like') # Earthlike
        ax[0].plot(list_zeng_masses,list_zeng_mg_radii,linewidth=zeng_width,color='grey',alpha=0.75,zorder=-100)#, label='100% Mantle')  # 100% Mantle
        line_swe_g, = ax[0].plot(
            x, radius_m(x,init_wmf_swe,init_teq_swe,init_age_swe),lw=0.75,color='teal',linestyle='-.',zorder=-100)#,label='50% Steam')
        line_swe_g100, = ax[0].plot(
            x, radius_g(x,init_wmf_swe100,init_teq_swe,init_age_swe),lw=0.75,color='teal',zorder=-100)#,label='100% Steam')
        # ax[0].legend(title=r'$H_2O$ $\dot{M}$ (Broome et al. 2025)', loc='lower right')
        ax[0].legend(loc='lower right', frameon=False)
        mdot_text = 'M' + '\u0307'
        # ax[0].text(1,3.45, f'H\u2082O Escape Rate \n(Broome et al. 2025)', fontsize='large', weight='semibold',ha='left', va='bottom')


        ymin = min(HHe_fin_rp)/R_e - 0.25
        ymax = max(HHe_fin_rp)/R_e + 0.5
        # ax[1].set_title("H-He Evolution (Owen & Wu 2017)")
        ax[1].set_xlabel(r'Mass [$M_\oplus$]')
        ax[1].set(xlim=(0.25,20.25), ylim=(ymin,ymax), ylabel=r'Radius [$R_\oplus$]')
        ax[1].scatter(init_mp/M_e, init_rp/R_e, c='#D13242', alpha=1, s=15, edgecolor='none', zorder=50, label='Initial state')
        ax[1].scatter(HHe_fin_mp/M_e, HHe_fin_rp/R_e, c='#76AA27', alpha=1, s=10, edgecolor='none', zorder=30, label='Final state')
        ax[1].plot(list_zeng_masses,list_zeng_ea_radii,linewidth=zeng_width,color='brown',alpha=0.75,zorder=-100)#,label='Earth-like') # Earthlike
        ax[1].plot(list_zeng_masses,list_zeng_mg_radii,linewidth=zeng_width,color='grey',alpha=0.75,zorder=-100)#, label='100% Mantle')  # 100% Mantle
        line_swe_g, = ax[1].plot(
            x, radius_m(x,init_wmf_swe,init_teq_swe,init_age_swe),lw=0.75,color='teal',linestyle='-.',zorder=-100)#,label='50% Steam')
        line_swe_g100, = ax[1].plot(
            x, radius_g(x,init_wmf_swe100,init_teq_swe,init_age_swe),lw=0.75,color='teal',zorder=-100)#,label='100% Steam')


        # ax[1].text(12,1, 'H-He Escape Rate \n(Owen and Wu 2017)', fontsize='large', weight='semibold',ha='left', va='bottom')
        ax[1].legend(loc='lower right', frameon=False)
        plt.savefig(savepath+'stacked_mr_trails'+identifier+'.png', bbox_inches='tight', pad_inches=0.5, dpi=600)
    
    # ------------------------------------------------------------------------------------------------ #
    
    if kde:
        import requests
        import datetime
        import seaborn as sns
        M_e = 5.97237e24 #Earth mass (kg)
        R_e = 6.371e6 #Earth radius (m)
        plt.figure(figsize=[8,6])


        # loop import planets starting and ending 
        mp_start = np.array([])
        rp_start = np.array([])
        mp_end = np.array([])
        rp_end = np.array([])

        datafolder = './evolution-data/paper/5000_xp/'+escapetype
        # period_gp = np.loadtxt(datafolder+'/5000_xp_gp_data.csv',skiprows=(1),usecols=(11),unpack=True)

        for i in range(n_planets):
            planetnum = i+1
            # import a planet's evolutionary track (just mp_gp and rp_gp)
            filename = f'{datafolder}/planet_{planetnum}-{n_planets}_evolution_data.csv'
            mp_gp,rp_gp = np.loadtxt(filename,usecols=(1,2),unpack=True)

            mp_start = np.append(mp_start, mp_gp[0])
            rp_start = np.append(rp_start, rp_gp[0])
            mp_end = np.append(mp_end, mp_gp[-2])
            rp_end = np.append(rp_end, rp_gp[-2])



        # ------------------------------------------------ #
        # UPDATE CATALOGS
        
        # NASA Exoplanet catalog
        def check_internet_connection():
            """Check if there is internet access by pinging a known URL."""
            try:
                requests.get("https://www.google.com", timeout=5)
                return True
            except requests.ConnectionError:
                return False

        def update_nea_exoplanet_catalog(catalog_url, output_file):
            """
            Updates the NEA exoplanet catalog from the NASA Exoplanet Archive TAP.
            Parameters:
                catalog_url (str): The TAP URL with the SQL query for the catalog.
                output_file (str): Path to save the downloaded catalog.
            """
            if not check_internet_connection():
                print("No internet connection. Unable to update the exoplanet catalog.")
                return

            try:
                response = requests.get(catalog_url, timeout=10)
                response.raise_for_status()  # Raise an error for HTTP issues
                
                # Prepare the header
                header = [
                    "# NASA Exoplanet Catalog",
                    f"# Source: {catalog_url}",
                    f"# Catalog last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
                
                # Write the header and data to the file
                with open(output_file, "w") as f:
                    f.write("\n".join(header) + "\n")
                    # Add '#' to the parameter names
                    data_lines = response.text.splitlines()
                    f.write("# " + data_lines[0] + "\n")  # Add # to parameter names
                    f.write("\n".join(data_lines[1:]) + "\n")
                
                print(f"Catalog updated successfully and saved to {output_file}")
            except requests.RequestException as e:
                print(f"Error fetching the catalog: {e}")

        # File paths and URL of the NEA Catalog
        nea_catalog_url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
                    "query=select+pl_name,pl_rade,pl_radeerr1,pl_radeerr2,"
                    "pl_masse,pl_masseerr1,pl_masseerr2,pl_eqt+from+ps+where+"
                    "default_flag=1+and+pl_controv_flag=0+and+pl_rade+is+not+null+"
                    "and+pl_masse+is+not+null+and+pl_bmassprov='Mass'&format=tsv")
        nea_output_file = "./Models/MARDIGRAS/data/catalog_exoplanets.dat"

        update_nea_exoplanet_catalog(nea_catalog_url, nea_output_file)


        # PlanetS catalog
        def update_planets_exoplanet_catalog(catalog_url, output_file):
            """
            Updates the PlanetS exoplanet catalog from the DACE website.
            Warning: Work in progress, currently not working
            """
            if True:
                print("PlanetS auto update currently not implemented, skipping updating attempt.")
                return

            if not check_internet_connection():
                print("No internet connection. Unable to update the exoplanet catalog.")
                return

            try:
                response = requests.get(catalog_url, timeout=10)
                response.raise_for_status()  # Raise an error for HTTP issues
                
                # Prepare the header
                header = [
                    "# NASA Exoplanet Catalog",
                    f"# Source: {catalog_url}",
                    f"# Catalog last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
                
                # Write the header and data to the file
                with open(output_file, "w") as f:
                    f.write("\n".join(header) + "\n")
                    # Add '#' to the parameter names
                    data_lines = response.text.splitlines()
                    f.write("# " + data_lines[0] + "\n")  # Add # to parameter names
                    f.write("\n".join(data_lines[1:]) + "\n")
                
                print(f"Catalog updated successfully and saved to {output_file}")
            except requests.RequestException as e:
                print(f"Error fetching the catalog: {e}")

        def read_planets_last_update(output_file):
            """
            Reads the date of the last update from the catalog file and prints it.
            Parameters:
                output_file (str): Path to the catalog file.
            """
            if not os.path.exists(output_file):
                print("Catalog file does not exist.")
                return

            try:
                with open(output_file, "r") as f:
                    for line in f:
                        text_lookup = "# Catalog last updated:"
                        if line.startswith(text_lookup):
                            print("# PlanetS catalog last updated: ",line.replace(text_lookup,'').replace('\n',''))
                            break
            except Exception as e:
                print(f"Error reading the catalog: {e}")

        # File paths and URL of the PlanetS Catalog
        planets_catalog_url = ("None")
        planets_output_file = "./Models/MARDIGRAS/data/catalog_exoplanets_planets.dat"

        # Update PlanetS catalog
        if not check_internet_connection():
            # Update the catalog
            update_planets_exoplanet_catalog(planets_catalog_url, planets_output_file)
        else:
            # Check for existing catalog and print the last update date
            read_planets_last_update(planets_output_file)


        # ------------------------------------------------ #
        # Load exoplanet catalog
        if catalog == "NEA":
            # Load NEA data
            data_nea = np.genfromtxt(nea_output_file,delimiter="\t",unpack=True,usecols=(1,2,3,4,5,6),filling_values=0.0)

            list_catalog_rp = data_nea[0]
            list_catalog_rpe1 = data_nea[1]
            list_catalog_rpe2 = data_nea[2]
            list_catalog_mp = data_nea[3]
            list_catalog_mpe1 = data_nea[4]
            list_catalog_mpe2 = data_nea[5]

            # Load the exoplanet names
            list_catalog_names = np.genfromtxt(
                nea_output_file, delimiter="\t", dtype=str, usecols=0
            )

            # Remove lines with empty Mp, Rp, or mass precision less than 50%
            exo_mass_prec = 0.5
            filter_nea = (list_catalog_rp!=0) & (list_catalog_mp!=0.0) \
            & ( abs(list_catalog_mpe1)/list_catalog_mp < exo_mass_prec) \
            & ( abs(list_catalog_mpe2)/list_catalog_mp < exo_mass_prec)

            list_exo_rp = list_catalog_rp[filter_nea]
            list_exo_rpe1 = list_catalog_rpe1[filter_nea]
            list_exo_rpe2 = list_catalog_rpe2[filter_nea]
            list_exo_mp = list_catalog_mp[filter_nea]
            list_exo_mpe1 = list_catalog_mpe1[filter_nea]
            list_exo_mpe2 = list_catalog_mpe2[filter_nea]
            list_exo_names = list_catalog_names[filter_nea]


        elif catalog == "PlanetS":
            # Load PlanetS data
            data_planets = np.genfromtxt(planets_output_file,delimiter="\t",unpack=True,usecols=(1,2,3,4,5,6),filling_values=0.0)

            list_catalog_rp = data_planets[0]
            list_catalog_rpe1 = data_planets[1]
            list_catalog_rpe2 = data_planets[2]
            list_catalog_mp = data_planets[3]
            list_catalog_mpe1 = data_planets[4]
            list_catalog_mpe2 = data_planets[5]

            # Load the exoplanet names
            list_catalog_names = np.genfromtxt(
                planets_output_file, delimiter="\t", dtype=str, usecols=0
            )

            # Remove lines with empty Mp and Rp
            exo_mass_prec = 0.5
            filter_planets = (list_catalog_rp!=0) & (list_catalog_mp!=0.0)

            list_exo_rp = list_catalog_rp[filter_planets]
            list_exo_rpe1 = list_catalog_rpe1[filter_planets]
            list_exo_rpe2 = list_catalog_rpe2[filter_planets]
            list_exo_mp = list_catalog_mp[filter_planets]
            list_exo_mpe1 = list_catalog_mpe1[filter_planets]
            list_exo_mpe2 = list_catalog_mpe2[filter_planets]
            list_exo_names = list_catalog_names[filter_planets]



        # ------------------------------------------------ #
        # define parameters and plot lines
        xmin, xmax, nx = 0.5, 35, 50 # 0.5, 20
        ymin, ymax  = 0.75, 4.25
        # xmin, xmax     = 2, 13
        # ymin, ymax     = 1, 3.25
        x = np.logspace(np.log10(xmin), np.log10(xmax), nx)

        #A25 lines
        line_swe_g, = plt.plot(
            x, radius_m(x,init_wmf_swe,init_teq_swe,init_age_swe),lw=0.75,color='teal',linestyle='-.',zorder=-100)#,label='50% Steam')
        line_swe_g100, = plt.plot(
            x, radius_g(x,init_wmf_swe100,init_teq_swe,init_age_swe),lw=0.75,color='teal',zorder=-100)#,label='100% Steam')

        # Zeng fixed
        zeng_width  = 1.0
        # plt.plot(list_zeng_masses,list_zeng_fe_radii,linewidth=zeng_width,color='black',zorder=5)
        plt.plot(list_zeng_masses,list_zeng_ea_radii,linewidth=zeng_width,color='brown',alpha=0.75,zorder=6)
        plt.plot(list_zeng_masses,list_zeng_mg_radii,linewidth=zeng_width,color='grey',alpha=0.75,zorder=7)
        # plt.plot(zeng_mass,zeng_50wat,linewidth=zeng_width,color='darkturquoise',alpha=0.75,zorder=-8)
        # plt.plot(zeng_mass,zeng_100wat,linewidth=zeng_width,color='darkturquoise',alpha=0.75,zorder=-9)


        # ------------------------------------------------ #
        # ADD SOLAR SYSTEM

        # Data for masses and radii of planets (in Earth masses and Earth radii)
        ss_planet_data = {
            "Venus": (0.815, 0.949),
            "Earth": (1.0, 1.0),
            "Uranus": (14.6, 4.01),
            "Neptune": (17.2, 3.88),
        }

        # Alchemy symbols for planets
        ss_alchemy_symbols = {
            "Venus": "♀",
            "Earth": "⊕",
            "Uranus": "♅",
            "Neptune": "♆",
        }

        # Extracting data
        ss_planets = list(ss_planet_data.keys())
        ss_masses = [ss_planet_data[planet][0] for planet in ss_planets]
        ss_radii = [ss_planet_data[planet][1] for planet in ss_planets]
        ss_symbols = [ss_alchemy_symbols[planet] for planet in ss_planets]

        for i in range(len(ss_planets)):
            text = plt.text(ss_masses[i], ss_radii[i], ss_alchemy_symbols[ss_planets[i]], \
                    fontsize=15, ha='center', va='center', color='black')
            #text.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])
        

        # ------------------------------------------------ #
        # KDE PLOT

        i_color = sns.cm.crest
        f_color = sns.cm.flare

        # unicorn_killer = period_gp > 1

        # Plot KDE for Dataset 1
        sns.kdeplot(x=mp_start/M_e, y=rp_start/R_e, cmap=i_color, log_scale=True, \
                    label="Initial", levels=5)
        sns.kdeplot(x=mp_start/M_e, y=rp_start/R_e, log_scale=True, fill=True, \
                    cmap=i_color, levels=5, alpha=0.25, legend=True)#, cbar=True)

        # Plot KDE for Dataset 2 #log_scale=True, 
        sns.kdeplot(x=mp_end/M_e, y=rp_end/R_e, cmap=f_color, log_scale=True, \
                    label="Final", levels=5)
        sns.kdeplot(x=mp_end/M_e, y=rp_end/R_e, log_scale=True, fill=True, \
                    cmap=f_color, levels=5, alpha=0.25,legend=True)#, cbar=True)


        # Planets
        list_exo_mpe = [abs(list_exo_mpe2), list_exo_mpe1]
        list_exo_rpe = [abs(list_exo_rpe2), list_exo_rpe1]

        # Exoplanets from the catalog file
        catalog_points = plt.errorbar(list_exo_mp,list_exo_rp,
                    yerr=list_exo_rpe,
                    xerr=list_exo_mpe,
                    fmt='o',zorder=-10,elinewidth=1,
                    c="slategray",label='Exoplanets',alpha=0.15)

        # plt.axhline(1.8, color='black', linestyle='--', linewidth=.75) #1.8 R_e radius valley


        # ------------------------------------------------ #
        # Add labels and title
        plt.title(f"Simulated {escapetype} Escape: {int(n_planets)} Steam Worlds", fontsize=15)
        plt.xlabel(r'Mass [$M_\oplus$]')
        plt.ylabel(r'Radius [$R_\oplus$]')
        plt.xscale('log')
        plt.yscale('linear')
        plt.xticks(ticks=[1,3,5,10,20,30],labels=['1','3','5','10','20','30'])
        plt.yticks(ticks=[1,2,3,4],labels=['1','2','3','4'])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.legend(fontsize=7,frameon=False)

        # import matplotlib.patches as mpatches
        # handles = [mpatches.Patch(facecolor=sns.cm.crest(100), label="Initial"),
        #            mpatches.Patch(facecolor=sns.cm.flare(100), label="Final")]  #sns.cm.crest(100)   sns.cm.flare(100)

        # plt.legend(handles=handles)#, fontsize=7)

        filename = f'{n_planets}_{escapetype}_kde'
        plt.savefig(f'./plots/paper/{filename}.png',bbox_inches='tight', dpi=600)
