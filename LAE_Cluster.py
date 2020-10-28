import numpy as np
from decimal import *
import math
getcontext().prec = 28


#Cosmology constants (not changeable)
#sigma_alpha = 5.9e-18 # T = 10^4 K
#sigma_alpha = 1.227*10**-16 # T = ???
#sigma_alpha = 3.73e-16 #T = 2.5 K
#sigma_alpha = 5.9e-16 # T = 1 K
rho_crit = 8.62*10**-27
mass_H = 1.672*10**-27
c = 3*10**8
H0 = float(68)/float(3.086e19)
OMm = 0.31
OMl = 0.75
baryon2DMfrac = 0.1
Hy2baryonfrac = 0.75
HyIGMfrac = 1
lya_min = 2.5e42
f12 = 0.4162
e = 1.602e-19
me = 9.11e-31
kb = 1.38064852e-23 #m^2 kg s^-2 K ^-1
mp = 1.672e-27 #kg
c = 3.0e8 #m/s
f_alpha_0 = 2.466e15 #1/s



class LAE_Cluster:
    #this code takes x direction as the line of sight
    def __init__(self, LAEpos, HII_DIM, DIM, L, **kwargs):
        self.LAEpos = LAEpos
        
        #geometric attributes
        self.HII_DIM = HII_DIM
        self.DIM = DIM
        self.L = L
        self.pix2com = float(self.L)/float(self.HII_DIM)
        self.V = L**3
        
        print('There are ' + str(LAEpos.shape[0]) + ' halos in this object')
        
        if 'slabs' in kwargs:
            self.slabs = slabs
            try:
                self.pixelsperslab = pixelsperslab
            except:
                return 'Please provide pixelsperslab'


    def list2box_3D(self, **kwargs):
        
        if 'M_alpha_min' in kwargs:
            M_alpha_min = kwargs.get('M_alpha_min')
        else:
            raise SyntaxError('kwargs : M_alpha_min missing')
        
        if 'slabs' in kwargs:
            print('SLABS !' , self.slabs)
            print('Using SLABS!!!!')
            LAE_Position_Box = np.zeros((self.slabs, self.HII_DIM, self.HII_DIM))
            LAE_Luminosity_Box = np.zeros((self.slabs, self.HII_DIM, self.HII_DIM))
            for i in range(self.LAEpos.shape[0]):
                slab_number = self.slabs
                while(slab_number >=0):
                    if (self.LAEpos[i][1]*HII_DIM >=slab_number*self.pixelsperslab) and (self.LAEpos[i][1]*self.HII_DIM < int(slab_number+1)*self.pixelsperslab):
                        y ,z = int(np.round(self.HII_DIM*self.LAEpos[i][2],0)), int(np.round(self.HII_DIM*self.LAEpos[i][3],0))
                        if y == self.HII_DIM or z == self.HII_DIM:
                            y,z = self.oops_we_are_at_the_edge(int(y),int(z))
                        LAE_Position_Box[slab_number][y][z] += 1
                        LAE_Luminosity_Box[slab_number][y][z]  += self.LAEpos[i][0]
                        break
                    slab_number = slab_number -1
            return LAE_Position_Box , LAE_Luminosity_Box

        else:
            LAE_Position_Box = np.zeros((self.HII_DIM, self.HII_DIM, self.HII_DIM))
            LAE_Luminosity_Box = np.zeros((self.HII_DIM, self.HII_DIM, self.HII_DIM))
            for i in range(self.LAEpos.shape[0]):
                Mh , x, y, z = float(self.LAEpos[i][0]), np.round(self.HII_DIM*self.LAEpos[i][1],0), np.round(self.HII_DIM*self.LAEpos[i][2],0), np.round(self.HII_DIM*self.LAEpos[i][3],0)
                if y == self.HII_DIM or z == self.HII_DIM:
                    y, z = self.oops_we_are_at_the_edge(int(y), int(z) )
                if x == self.HII_DIM or y == self.HII_DIM:
                    x, y = self.oops_we_are_at_the_edge(int(x), int(y) )
                LAE_Position_Box[int(x)][int(y)][int(z)] += 1
                LAE_Luminosity_Box[int(x)][int(y)][int(z)] += float(float(Mh)*lya_min)/float(M_alpha_min)
            return LAE_Position_Box , LAE_Luminosity_Box


    @staticmethod
    def reionization_usage():
        #be careful of the double print thing
        return 'If applying a reionization history, reionization = [z_start, z_end, N, nboxes, directory]'
    
    @staticmethod
    def apply_parameters_usage():
        return 'If filtering through the list, provide args = [fduty, Malphamin]'
    
    def oops_we_are_at_the_edge(self, y,z):
        if y == self.HII_DIM and z == self.HII_DIM:
            return int(self.HII_DIM - 1), int(self.HII_DIM - 1)
        else:
            if y == self.HII_DIM:
                return int(self.HII_DIM - 1) , z
            if z == self.HII_DIM:
                return y, int(self.HII_DIM - 1)

    def rand_choice(self, fduty, list_to_filter):
        choices = (0, 1)
        choices = np.array(choices)
        p = (1-fduty, fduty)
        p = np.array(p)
        #print('SHAPE of LIST TO FILER0', list_to_filter.shape[0])
        #DELETETHIS = np.random.choice(choices, list_to_filter.shape[0], p = p)
        #print("RETURNING " + str(DELETETHIS.shape))
        return np.random.choice(choices, list_to_filter.shape[0], p = p)



    #************************************** TATISOMI APPROXIMATION ************************************
    @staticmethod
    def compute_sigma(H, delta_vd):
        f12 = 0.4162
        e = 1.602e-19
        me = 9.11e-31
        c = 3e8
        return H*(float(f12*np.sqrt(np.pi)*e*e)/float(me*c*delta_vd))
    
    @staticmethod
    def compute_alpha(delta_vd):
        delta_vl = 9.936e7 #1/s
        return float(delta_vl)/float(2*delta_vd)
    
    @staticmethod
    def compute_H(q, x):
        return (float(np.sqrt(np.pi)))*(q + float(np.exp(-1*(x**2)))/float(1.77245385))
    
    @staticmethod
    def f2z(f, f0):
        return ((float(f0)/float(f)) -1)
    
    @staticmethod
    def z2f(z, f0):
        return float(float(f0)/float((1 + float(z))))
    
    @staticmethod
    def z2f_Decimal(z, f0):
        f0 = Decimal(2.466e15)
        z = Decimal(z)
        return Decimal(Decimal(f0)/Decimal((Decimal(1) + Decimal(z))))
    
    @staticmethod
    def compute_x(f, f0, delta_vd):
        return float(float(f) - float(f0))/float(delta_vd)
    
    @staticmethod
    def compute_q(alpha, x):
        #check if it is 0
        zed = float(x**2 - 0.855)/float(x**2 + 3.42)
        if zed <= 0:
            return 0 , zed
        else:
            return  ((zed*(1 + float(21)/float(x**2)))*(float(alpha)/float(np.pi*x**2 +np.pi))*(0.1117 + zed*(4.421+zed*(-9.207 +5.674*zed)))) , zed

    @staticmethod
    def compute_delta_vd(T, f0):
        kb = 1.38064852e-23 #m^2 kg s^-2 K ^-1
        mp = 1.672e-27 #kg
        c = 3.0e8 #m/s
        f_alpha_0 = 2.466e15 #1/s
        return f0*np.sqrt(float(2*kb*T)/float(mp*c*c))

    #************************************** END TATISOMI APPROXIMATION ************************************
    
    @staticmethod
    def compute_a_nu(T):
        return float(4.7e-4)*np.sqrt(float(10**4)/float(T))
    @staticmethod
    def compute_nu_thermal(T):
        return np.sqrt(float(T*2*1.38e-16)/float(1.66e-24))
    @staticmethod
    def compute_delta_nu_alpha(nu_thermal):
        return float((2.466e15)*nu_thermal)/float(3e10)
    @staticmethod
    def compute_x_frac(nu, delta_nu_alpha):
        return float(float(nu) - float(2.466e15))/float(delta_nu_alpha)
    @staticmethod
    def compute_nu_at_z(z):
        return float(2.466e15)/float(float(1) +float(z))
    @staticmethod
    def compute_x_at_first_pixel(T, delta_z):
        return np.abs(float(float(1)/float(float(1) + float(delta_z))) - float(1))/float(float(np.sqrt( float(T*2*1.38e-16)/float(1.66e-24)))/float(3.0e10))

    @staticmethod
    def compute_sigma_alpha_avg(T,x_intersection, x_cutoff, a_nu):
        return float(5.9e-18)*np.sqrt(float(10**4)/float(T))*float( float(float(np.sqrt(np.pi)*math.erf(x_intersection))/float(2)) + float(float(a_nu)/float(np.sqrt(np.pi)))*( float(float(1)/float(x_intersection)) - float( float(1)/float(x_cutoff) ) ))
    @staticmethod
    def compute_phi_x_wing(a_nu, x):
        return float(a_nu)/float(x*x*np.sqrt(np.pi))

    @staticmethod
    def compute_phi_x_core(x):
        return np.exp(-x*x)


    def find_intersection(self, T):
        x = np.linspace(0.05, 5, 200)
        wing = np.zeros_like(x)
        core = np.zeros_like(x)
        diff = []
        a = self.compute_a_nu(T)
        for i in range(len(x)):
            wing[i] = self.compute_phi_x_wing(a, x[i])
            core[i] = self.compute_phi_x_core(x[i])
            diff.append(np.abs(core[i]-wing[i]))
        diff = np.array(diff)
        return x[np.where(diff == diff.min())]




    def tau_IGM_classic(self, x,y, z_range, lightcone):
        tau = 0
        z_i = z_range[0]
        z_end = z_range[len(z_range)-1]
        delta_z = float(np.max(z_range)-np.min(z_range))/float(lightcone.shape[0])
        ctr = 0
        #print('about to start with ' , z_i, ' and z_end' , z_end)
        while(z_i > z_end ):
            #print('doing z_i' , z_i, ' at ctr ' , ctr)
            tau += delta_z*float((sigma_alpha*OMm*rho_crit*baryon2DMfrac/(mass_H*H0))*(1+z_i)**2*(lightcone[ctr][x][y]))/float(np.sqrt(OMm*(1+z_i)**3 +OMl))
            #print('adding tau' , tau)
            ctr += 1
            z_i = z_range[ctr]
        #print('tau is' , tau)
        return tau



    def tau_IGM_avgcore_wing_new(self, x,y, z_range, lightcone,  sigma_alpha_0_neutral, sigma_alpha_0_ionized, sigma_alpha_effective_neutral, sigma_alpha_effective_ionized , delta_z, **kwargs):
        z_i = z_range[0]
        z_end = z_range[len(z_range)-1]
        tau = 0
        lightcone_redshifts = kwargs.get('lightcone_redshifts')
        density_lightcone = kwargs.get('density')

        
        for i in range(19):
            
            #the frequency doesn't care about what the temperature is
            nu = self.compute_nu_at_z((i+1)*delta_z)
            
            if i  == 0:
                #no need to use the voigt functino here since we've already averaged over it
                phi_x = 1
                HyIGMfrac = 1
                #we're at the first pixel, use the average cross section from x = 0 to the x at the end of the first pixel
                if lightcone[i][x][y] < 0.01:
                    #cell is ionized so use T_ionized
                    sigma_alpha_0 = sigma_alpha_effective_ionized
                else:
                    sigma_alpha_0   = sigma_alpha_effective_neutral
            else:
                if lightcone[i][x][y] < 0.01:
                    nu_thermal = self.compute_nu_thermal(self.T_ionized)
                    a_nu = self.a_nu_ionized
                    sigma_alpha_0 = sigma_alpha_0_ionized
     
                else:
                    nu_thermal = self.compute_nu_thermal(self.T_neutral)
                    a_nu = self.a_nu_neutral
                    sigma_alpha_0 = sigma_alpha_0_neutral
                
                HyIGMfrac = 0.2
                delta_nu_alpha = self.compute_delta_nu_alpha(nu_thermal)
                x_frac  =  self.compute_x_frac(nu,  delta_nu_alpha)
                phi_x = self.compute_phi_x_wing(a_nu, x_frac)
    
                    
            tau += delta_z*float((c*sigma_alpha_0*phi_x*OMm*np.abs(float(1) + float(density_lightcone[i][x][y]))*Hy2baryonfrac*rho_crit*HyIGMfrac*baryon2DMfrac/(mass_H*H0))*(1+z_i)**2*(lightcone[i][x][y]))/float(np.sqrt(OMm*(1+z_i)**3 +OMl))
                    
        return tau
        
            
            
    def sort_into_slabs(self, **kwargs):
        slabs = kwargs.get('slabs')
        pixelsperslab = kwargs.get('pixelsperslab')
        
        #make a new instance
        self.pixelsperslab = pixelsperslab
        
        #Slab0, Slab1, Slab2, Slab3, Slab4, Slab5, Slab6, Slab7 = ( [] for i in range(slabs) )
        #Slab_Dictionary = {'0': Slab0, '1':Slab1, '2':Slab2, '3':Slab3, '4': Slab4, '5':Slab5, '6':Slab6, '7':Slab7}
        
        #initialize dictionary with each entry a new slice
        Slab_Dictionary = {}
        for i in range(slabs):
            Slab_Dictionary[str(i)] = []
        for i in range(self.LAEpos.shape[0]):
            slab_number = slabs - 1
            while(slab_number >=0):
                if (np.round(self.LAEpos[i][1]*self.HII_DIM,0) >=slab_number*pixelsperslab) and (np.round(self.LAEpos[i][1]*self.HII_DIM,0) < int(slab_number+1)*pixelsperslab):
                    Slab_Dictionary[str(slab_number)].append((self.LAEpos[i][0], self.LAEpos[i][1], self.LAEpos[i][2], self.LAEpos[i][3] ))
                    break
                slab_number = slab_number -1
        for i in range(slabs):
            Slab_Dictionary[str(i)] = np.array(Slab_Dictionary[str(i)])
        #now that we are using slabs, modify the class attribute of volume
        self.V = self.pix2com * pixelsperslab * self.L**2
        return Slab_Dictionary


    def apply_parameters(self, *args,  **kwargs):
        filtered_list = []
        #check to see if we are using other paramters
        #print("HERE ARE THE ARGS " + str(args) )
        try:
            fduty = args[0][0]
            M_alpha_min = args[0][1]
        #print('FDUTY AND MALPHAMIN ARE' + str(fduty) , M_alpha_min)
        except:
            print('OOPS')
            self.apply_parameters_usage()
        
        if 'LAEpos' in kwargs:
            list_to_filter = kwargs.get('LAEpos')
        else:
            print('GOING WITH DEFAULT')
            list_to_filter = self.LAEpos

        if 'compare' in kwargs:
            are_we_comparing = kwargs.get('compare')
        else:
            are_we_comparing = False


        chi = self.rand_choice(fduty, list_to_filter)

        for i in range(list_to_filter.shape[0]):
            if (float(lya_min*list_to_filter[i][0]*chi[i])/float(M_alpha_min)) >=  lya_min:
                filtered_list.append((float(lya_min*list_to_filter[i][0]*chi[i])/float(M_alpha_min), list_to_filter[i][1], list_to_filter[i][2], list_to_filter[i][3]))

        
        if are_we_comparing == True:
            return np.array(filtered_list), chi
        else:
            return np.array(filtered_list)


    def apply_reionization_slabs(self, lightcone_Dictionary, z_range, T_neutral, T_ionized, **kwargs):
        
        #we need to assume a model for the gas an ionized temperature:
        #EoR attributes
        self.T_neutral = T_neutral #this corresponds to a neutral pixel
        self.T_ionized =  T_ionized #10**4 #this corresponds to an ionized pixel
        self.x_intersection_neutral, self.x_intersection_ionized = self.find_intersection(self.T_neutral), self.find_intersection(self.T_ionized)
        self.a_nu_neutral, self.a_nu_ionized = self.compute_a_nu(self.T_neutral), self.compute_a_nu(self.T_ionized)
        
        #are we using a different Halo list or the one that was already defined?
        filtered_list = []
        try:
            list_to_filter = kwargs.get('LAEpos')
        except:
            list_to_filter = self.LAEpos
        
        #check to see if we are given the redshifts which span the lightcone. If not assume the lightcone linearly spans the entire z_range.
        #In the future, change this to force the user to supply the redshits
        if 'lightcone_redshifts' in kwargs:
            lightcone_redshifts = kwargs.get('lightcone_redshifts')
            delta_z = np.abs(lightcone_redshifts[1] - lightcone_redshifts[0])
        else:
            #generating redshifts
            print('assuming the lightcone spans ' + str(z_range) + ' in '  + str(lightcone.shape[0])  + ' steps' )
            lightcone_redshifts = np.zeros((lightcone.shape[0]))
            delta_z = float(np.max(z_range)-np.min(z_range))/float(lightcone.shape[0])
            for i in range(lightcone.shape[0]):
                lightcone_redshifts[i] = z_range[0] - i*delta_z
    
        #load the overdensity profile to compute the number density of hydrogen within each pixel (in tau)
        if 'density' in kwargs:
            density_lightcone = kwargs.get('density')
        
        #in this optical depth model, we compute the average cross section from line center to the end of the first pixel in the lightcone.
        #we then use this average cross section to compute tau for that first pixel (before switching to the wing in subsequent pixels)
        #since by the end of the first pixel in the lightcone the cross section has likely switched from the core to the wing, when averaging sigma_alpha, we need to
        #know where that transition occured. Luckily, the intersection only depends on the temperature of the first pixel. Let us compute them here and send them to our
        #tau calculation.
        #We also need to know what is the boundary of the integral to average sigma. We refer to that as x_cutoff. The cutoff depends only temperature (thermal width of the line)
        #and the frequency of the lya photon at the end of the first pixel. The frequency of the lya photon is determined by the redshift displacement at the end of the first pixel.
        #Since the redshift spacing is already set from above, let's compute the generic frequency of the lya photon for both cases of a neutral and ionized pixel (to save some time)
        x_cutoff_neutral, x_cutoff_ionized = self.compute_x_at_first_pixel(self.T_neutral, delta_z), self.compute_x_at_first_pixel(self.T_ionized, delta_z)


        #before starting, we need to find what we will be using as the effective lya cross section for the first pixel
        #the average cross section doesn't have strong dependence on the temperature of the cell. However it does depend on what the z is at the end of the cell
        #so for a given delta z , all points will have approximately the same sigma_average, regardless of the temperature (check this)
        sigma_alpha_effective_neutral, sigma_alpha_effective_ionized = self.compute_sigma_alpha_avg(self.T_neutral, self.x_intersection_neutral, x_cutoff_neutral, self.a_nu_neutral), self.compute_sigma_alpha_avg(self.T_ionized, self.x_intersection_ionized, x_cutoff_ionized, self.a_nu_ionized)
        sigma_alpha_effective_neutral, sigma_alpha_effective_ionized = float(sigma_alpha_effective_neutral)/float(x_cutoff_neutral), float(sigma_alpha_effective_ionized)/float(x_cutoff_ionized)
        sigma_alpha_0_neutral, sigma_alpha_0_ionized = float(5.9e-18)*np.sqrt(float(10**4)/float(self.T_neutral)), float(5.9e-18)*np.sqrt(float(10**4)/float(self.T_ionized))



        for i in range(list_to_filter.shape[0]):
            x, y, z = np.round(self.HII_DIM*list_to_filter[i][1],0), np.round(self.HII_DIM*list_to_filter[i][2],0), np.round(self.HII_DIM*list_to_filter[i][3],0)
            if y == self.HII_DIM or z == self.HII_DIM:
                y, z = self.oops_we_are_at_the_edge(int(y),int(z))
            if x == self.HII_DIM:
                x = self.HII_DIM - 1
            #x = 187
            tau = self.tau_IGM_avgcore_wing_new(int(y), int(z), z_range, lightcone_Dictionary[int(x)], sigma_alpha_0_neutral, sigma_alpha_0_ionized, sigma_alpha_effective_neutral, sigma_alpha_effective_ionized , delta_z, lightcone_redshifts = lightcone_redshifts, density = density_lightcone[int(x)])

            if float(list_to_filter[i][0]*np.exp(-tau)) > lya_min:
                filtered_list.append((list_to_filter[i][0]*np.exp(-tau), list_to_filter[i][1], list_to_filter[i][2], list_to_filter[i][3]))
        
        return np.array(filtered_list)

    def map_slab2box(self, **kwargs):
        #SOUND the fucking alarm here, after we are returning the filtered LAEpos, does that change self.LAEpos?
        
        if 'binary' in kwargs:
            binary = kwargs.get('binary')
        else:
            binary = False
        
        try:
            list_to_map = kwargs.get('LAEpos')
        except:
            list_to_map = self.LAEpos

        if np.ndim(list_to_map) > 2:
            print('Sending this shit to map2box')
            self.map2box(LAEpos = list_to_map)
            
        else:
            LAE_Position_Box = np.zeros((self.HII_DIM, self.HII_DIM))
            LAE_Luminosity_Box = np.zeros((self.HII_DIM, self.HII_DIM))
            for i in range(list_to_map.shape[0]):
                y, z =  np.round(self.HII_DIM*list_to_map[i][2],0), np.round(self.HII_DIM*list_to_map[i][3],0)
                if y == self.HII_DIM or z == self.HII_DIM:
                    y, z = self.oops_we_are_at_the_edge(int(y),int(z))
                if binary:
                    LAE_Position_Box[int(y)][int(z)] = 1
                    LAE_Luminosity_Box[int(y)][int(z)] = list_to_map[i][0]
                else:
                    LAE_Position_Box[int(y)][int(z)] += 1
                    LAE_Luminosity_Box[int(y)][int(z)] += list_to_map[i][0]
            return LAE_Position_Box , LAE_Luminosity_Box


    def map2box(self, **kwargs):
    #SOUND the fucking alarm here, after we are returning the filtered LAEpos, does that change self.LAEpos?
        try:
            list_to_map = kwargs.get('LAEpos')
        except:
            list_to_map = self.LAEpos

        if np.ndim(list_to_map) == 2:
            map_slab2box(LAEpos = list_to_map)
        
        else:
            print('LIST TO MAP SHAPE ' + str(list_to_map.shape[0]))
            LAE_Position_Box = np.zeros((self.HII_DIM, self.HII_DIM, self.HII_DIM))
            LAE_Luminosity_Box = np.zeros((self.HII_DIM, self.HII_DIM, self.HII_DIM))
            for i in range(list_to_map.shape[0]):
                x, y, z = np.round(self.HII_DIM*list_to_map[i][1],0), np.round(self.HII_DIM*list_to_map[i][2],0), np.round(self.HII_DIM*list_to_map[i][3],0)
                if y == self.HII_DIM or z == self.HII_DIM:
                    y, z = self.oops_we_are_at_the_edge(int(y),int(z))
                if x == self.HII_DIM or z == self.HII_DIM:
                    x, z = self.oops_we_are_at_the_edge(int(x),int(z))
                LAE_Position_Box[int(x)][int(y)][int(z)] += 1
                LAE_Luminosity_Box[int(x)][int(y)][int(z)] += list_to_filter[i][0]
            return LAE_Position_Box , LAE_Luminosity_Box


    def density(self, **kwargs):
        try:
            list_of_LAEs = kwargs.get('LAEpos')
        except:
            list_of_LAEs = self.LAEpos

        return float(list_of_LAEs.shape[0])/float(self.pix2com * self.pixelsperslab * self.L**2)


    def remove_los_from_list(self, **kwargs):
        try:
            list_of_LAEs = kwargs.get('LAEpos')
        except:
            list_of_LAEs = self.LAEpos
        filtered_list = []
        for i in range(list_of_LAEs.shape[0]):
            filtered_list.append((self.HII_DIM*list_of_LAEs[i][2], self.HII_DIM*list_of_LAEs[i][3]  ))

        return np.array(filtered_list)

        
    def search_box_for_repeats(self, **kwargs):
        box = kwargs.get('box')
        repeats = 2
        repeat_list = []
        print('the max in the box is ', np.max(LAE_box))
        while(repeats < np.max(box)):
            matches = [x for x in flatten_oi if x == repeats]
            repeat_list.append(len(matches))
            repeats += 1
        return repeat_list
    
            
    def check_repeats(self, **kwargs):
        try:
            list_of_LAEs = kwargs.get('LAEpos')
            if list_of_LAEs.shape[0] == self.HII_DIM and list_of_LAEs.shape[1] == self.HII_DIM:
                matches = search_box_for_repeats(box = list_of_LAEs)
                return matches
        except:
            list_of_LAEs = self.LAEpos

        box = self.map2box(LAEpos = list_of_LAEs)
        matches = search_box_for_repeats(box = box)
        return matches

    def extract_luminosities(self, **kwargs):
        print('WARNING : This method assumes the luminosity parameters have already been applied')
        if 'LAEpos' in kwargs:
            list_to_filter  = kwargs.get('LAEpos')
        else:
            list_to_filter = self.LAEpos

        luminosities = []
        for i in range(list_to_filter.shape[0]):
            luminosities.append(list_to_filter[i][0])
        return np.array(luminosities)
        










