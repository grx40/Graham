import numpy as np
import matplotlib.pyplot as pl
#import angular_correlation_function as acf
import sys
from decimal import *
from numpy.fft import fft, fftfreq, ifft, fft2, fftshift, fftn
from subprocess import *
import os
import emcee
import compute_ps_updated_beta as compute_ps_updated
from emcee.utils import MPIPool
import lightcone_LAEinterface_decreasingz_xlos as lc
import astropy
import LAE_Cluster as LAEC
import xi_2D
import time

#let's name this run 
RUN= int(np.abs(np.random.uniform(0, 2000)))
os.system("echo RUN IS : " + str(RUN))


def find_foldername(redshift, OUTPUT_NUMBER,  mode = 'some type'):
    #call(["echo looking for a " + str(mode) + " box at redshift " + str(redshift) ], shell = False )
    filename_list = open(boxes_path +'Output_files/filename_list_'+ str(mode) +'_' + str(OUTPUT_NUMBER) , 'w')
    call(["ls " + boxes_path + "*" + str(OUTPUT_NUMBER)], stdout = filename_list, shell = True)
    filename_list.close()
    with open(boxes_path + 'Output_files/filename_list_' + str(mode) + '_' + str(OUTPUT_NUMBER), 'r') as inF:
        for line in inF:
            if mode in line and str(float(redshift)) in line and str(OUTPUT_NUMBER) in line:
                return  line.replace("\n", '')


def copy_FROM_TO(FROM_OUTPUT_NUMBER, TO_OUTPUT_NUMBER):
#this does all boxes that have the desired tag
    filename_list = open(boxes_path +'Output_files/filename_list_'+str(FROM_OUTPUT_NUMBER)+'_' +str(TO_OUTPUT_NUMBER), 'w')
    call(["ls ../Boxes/*"  + str(FROM_OUTPUT_NUMBER)], stdout = filename_list, shell = True)
    filename_list.close()
    with open(boxes_path + 'Output_files/filename_list_'+str(FROM_OUTPUT_NUMBER)+'_' +str(TO_OUTPUT_NUMBER) , 'r') as inF:
        for line in inF:
            os.system("cp " + line.replace("\n", '') + " " + line.replace(str(FROM_OUTPUT_NUMBER),str(TO_OUTPUT_NUMBER)))
    

def moveout_xH(OUTPUT_NUMBER):
    filename_list = open(boxes_path +'Output_files/filename_list_'+str(OUTPUT_NUMBER)+ '_out', 'w')
    call(["ls ../Boxes/*"  + str(OUTPUT_NUMBER)], stdout = filename_list, shell = True)
    filename_list.close()
    with open(boxes_path + 'Output_files/filename_list_'+str(OUTPUT_NUMBER) + '_out' , 'r') as inF:
        for line in inF:
            if 'xH' in line and str(OUTPUT_NUMBER) in line:
                os.system("mv " + str(line.replace("\n",''))+ " ../Boxes/ionizationfield_NoDecorrelation/")


def movein_xH(OUTPUT_NUMBER):
    filename_list = open(boxes_path +'Output_files/filename_list_'+str(OUTPUT_NUMBER)+ '_in', 'w')
    call(["ls ../Boxes/ionizationfield_NoDecorrelation/*"  + str(OUTPUT_NUMBER)], stdout = filename_list, shell = True)
    filename_list.close()
    with open(boxes_path + 'Output_files/filename_list_'+str(OUTPUT_NUMBER) + '_in' , 'r') as inF:
        for line in inF:
            if 'xH' in line and str(OUTPUT_NUMBER) in line:
                os.system("mv " + str(line.replace("\n", ''))  +  " ../Boxes/")


def delete_xH(OUTPUT_NUMBER):
#this moves all  xH_noDecor to a safe temporary location
    filename_list = open(boxes_path +'Output_files/filename_list_xHdelete_'+str(OUTPUT_NUMBER), 'w')
    call(["ls ../Boxes/*"  + str(OUTPUT_NUMBER)], stdout = filename_list, shell = True)
    filename_list.close()
    with open(boxes_path + 'Output_files/filename_list_xHdelete_'+str(OUTPUT_NUMBER) , 'r') as inF:
        for line in inF:
            if 'xH' in line and str(OUTPUT_NUMBER) in line:
                os.system("rm " + line.replace("\n", ''))


def load_data(path, HII_DIM):
    data = np.fromfile(path ,dtype=np.float32)
    return data.reshape((HII_DIM,HII_DIM,HII_DIM))
    
def beta2sigma(beta):
    if beta >= 0:
        sign = np.sign(-np.pi*(beta) + np.pi)
        sigma = np.abs(-np.pi*(beta) + np.pi)
        return (sign*sigma)
    else:
        sign = np.sign(-np.pi*(beta) - np.pi)
        sigma = np.abs(-np.pi*(beta) - np.pi)
        return (sign*sigma)

####################################################################
#          Cosmology and Astrophysical Parameters                 #
####################################################################

#EoR parameters (fixed in this version)
zeta = 500
Mturn = 10
Rmfp = 30
M_alpha_min_fiducial = 1e10

#Cosmology constants
sigma_alpha = 1.227*10**-16
rho_crit = 8.62*10**-27
mass_H = 1.672*10**-27
c = 3*10**8
nHI = 10
H0 = float(68)/float(3.086e19)
OMm = 0.25
OMl = 0.75
baryon2DMfrac = 0.05
T_neutral = 1
T_ionized = 10**4


####################################################################
#                    Emcee specific parameters                    #
####################################################################

#os.chdir("/home/grx40/projects/def-acliu/grx40/soft/21cmFASTM/Programs/")
#dimensions and walkers of EnsembleSampler
ndim = 6
nwalkers = 12



####################################################################
#                     LAE script parameters               #
####################################################################
#Constants for the script
pixelsperslab = 25
Mpcperslab = 37.5
Box_length = 300
lya_min = 2.5e42
HII_DIM = 200
DIM = 800

z_starts = (8.0, 7.5, 7.0, 6.5)
z_end = 6.0
N = 150
nboxes = (5,4,3,2)
slabs = 2

#Load the halo file that we will be using as a template (This is possiby the master halo list). Load this only once for the entire run
#and load all the redshifts into a single array
#also, make an instance of the LAE cluster for each redshift and sort them into the 8 slabs. store the results into arrays that span redshift space and do this only once per run
halopos_z = np.zeros((len(z_starts)), dtype = object)
LAE_positions_z = np.zeros((len(z_starts)), dtype = object)
LAE_slabs_z = np.zeros((len(z_starts)), dtype = object)

for i in range(len(z_starts)):
    starts = 6.6
    box = 'halos_z'+str(np.round(starts,1))+'0_800_300Mpc_82600201'
    halopos_z[i] = np.genfromtxt( box , dtype=None)
    LAE_positions_z[i] = LAEC.LAE_Cluster(halopos_z[i], HII_DIM, DIM, Box_length)
    LAE_slabs_z[i] = LAE_positions_z[i].sort_into_slabs(slabs = 8, pixelsperslab = 25)

#make acfbins for the model ACF
acfbins = xi_2D.create_k_boundaries(0.008, 1.3 , mode = 'custom', bins = np.linspace(1,50, 50))[1]


####################################################################
#                 Define Bayesian Probabilities                    #
####################################################################

#lets add the number density to the prior. If we constrain it using the likelihood then we may end up in the
#unfortunate situation of having the code get stuck with a gigantic ACF
def lnprior(x):
    beta = x[0]
    fduty = x[1]
    zeta = x[2]
    Mturn = x[3] 
    M_alpha_min = x[4]
    Rmfp = x[5]

    if  -1  < beta < 1 and 0 < np.exp(fduty) <= 1 and  200 < zeta < 1000 and 1e7 < (Mturn*5e7) < 9e9 and 1e9 < M_alpha_min < 1e11 and 5 < Rmfp < 60:
        os.system("echo RUN " + str(RUN) + " accepting the fuck out of beta and fduty " + str(beta) + " " + str(np.exp(fduty)) + " " + str(zeta) + " " + str(Mturn) +" " + str(M_alpha_min/float(1e10)) + " " + str(Rmfp)   )
        return 0.0
    os.system("echo RUN " + str(RUN) + " Rejecting the fuck out of beta and fduty " + str(n_density) + " " + str(beta) + " " + str(np.exp(fduty)) + " " + str(zeta) + " " + str(Mturn) +" " + str(M_alpha_min/float(1e10)) + " " + str(Rmfp ) )
    return -np.inf


def lnprob(x, ACF_fiducial_z, density_of_observable_LAEs_afterEoR_fiducial_z):
    lp = lnprior(x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(x, ACF_fiducial_z, density_of_observable_LAEs_afterEoR_fiducial_z)

beta_list = []
fduty_list = []
nmodel_list =[]
zeta_list = []
Mturn_list = []
M_alpha_min_list = []
Rmfp_list = []
acf_list = []
chi2_ndensity = []
chi2_acf = []
def lnlike(x, ACF_fiducial_z, density_of_observable_LAEs_afterEoR_fiducial_z):
    #draw a tag for this run
    OUTPUT_NUMBER = int(np.abs(np.random.uniform(1000000, 9990000)))
    
    #map emcee space to EoR parameters
    beta = x[0]
    fduty = np.exp(x[1])
    zeta = x[2]
    Mturn = x[3]
    M_alpha_min = x[4]
    Rmfp = x[5]

    beta_list.append(beta)
    fduty_list.append(fduty)
    zeta_list.append(zeta)
    Mturn_list.append(Mturn)
    M_alpha_min_list.append(M_alpha_min)
    Rmfp_list.append(Rmfp)

    if beta >= 0:
        sign = np.sign(-np.pi*(beta) + np.pi)
        sigma = np.abs(-np.pi*(beta) + np.pi)
    else:
        sign = np.sign(-np.pi*(beta) - np.pi)
        sigma = np.abs(-np.pi*(beta) - np.pi)
    
    
    t21_i = time.time()
    #make the reionization scenario for these parameters
    os.system("echo choice of beta is "  + str(beta) + ' leading to a sigma of' + str(sigma) +' with sign' + str(sign) )
    os.system("./init " + str(sign) + ' ' + str(sigma) +' ' + str(OUTPUT_NUMBER) )
    os.system("./drive_zscroll_noTs " + str(10*zeta) +' ' + str(Rmfp) +' ' + str(Mturn*5*10**7)+ ' '  + str(OUTPUT_NUMBER))
    t21_f = time.time()
    os.system("echo 21cmfast runtime is " + str(t21_f - t21_i))

    #let us make the lightcone for this reionization scenario
    xH_lightcone_z_dictionary = np.zeros((len(z_starts), len(betas)), dtype  = object)
    delta_lightcone_z_dictionary = np.zeros((len(z_starts), len(betas)), dtype  = object)
    redshifts_lightcone_z_dictionary = np.zeros((len(z_starts), len(betas)), dtype  = object)
    
    start = time.time()
    for z in range(len(z_starts)):
        #initialize a placeholder dictionary for this redshift
        lightcone_Dictionary = {}
        density_lightcone_Dictionary = {}
        
        for i in range(slabs*pixelsperslab):
            box_slice = HII_DIM - i -1
            lightcone_Dictionary[int(box_slice)] , lightcone_redshifts = lc.lightcone(DIM = HII_DIM, z_start = z_starts[z], z_end = z_end, N = N, nboxes = nboxes[z], box_slice = int(box_slice), directory = '../Boxes/', sharp_cutoff = 20, return_redshifts = True , tag = str(OUTPUT_NUMBER) )
            density_lightcone_Dictionary[int(box_slice)], lightcone_redshifts  = lc.lightcone(DIM = HII_DIM, z_start = z_starts[z], z_end = z_end, N = N, nboxes = nboxes[z], box_slice = int(box_slice), directory = '../Boxes/', marker = 'updated_smoothed' ,  sharp_cutoff = 20, return_redshifts = True , tag = str(OUTPUT_NUMBER) )
        
        os.system('echo Finished lightcone for redshift  ' + str(z_starts[z]))

        #save this dictionary into the array
        xH_lightcone_z_dictionary[z] = lightcone_Dictionary
        delta_lightcone_z_dictionary[z] = densitylightcone_Dictionary
        redshifts_lightcone_z_dictionary[z] = lightcone_redshifts

    end = time.time()
    os.system("RUNTIME OF LIGHTCONE (s) : " + str( end - start))

    #okay, we are ready to apply the scenario corresponding to these parameters onto the master list of halos
    #intrinsic luminosity params for this model are
    intrinsic_params = (fduty, M_alpha_min)

    #initialize observables
    ACF_model_z = np.zeros((len(z_starts) , len(acfbins)))
    density_of_observable_LAEs_afterEoR_z = np.zeros_like(z_starts)

    for z in range(len(z_starts)):

        for sl in range(slabs):
            #these are the observable LAEs without reionization
            Observable_LAEs = LAE_positions_z[z].apply_parameters(intrinsic_params, LAEpos = LAE_slabs_z[z][str(7-sl)] )

            #compute the density of observable LAEs without a reionization scenario
            density_of_observable_LAEs =  LAE_positions_z[z].density(LAEpos = Observable_LAEs)

            #apply fiducial reionization scenario (inside out)
            LAE_EoR = LAE_positions.apply_reionization_slabs(xH_lightcone_z_dictionary[z], z_range, T_neutral , T_ionized, lightcone_redshifts = redshifts_lightcone_z_dictionary[z], density =  delta_lightcone_z_dictionary[z],  LAEpos = Observable_LAEs)

            #compute the density after reionization model is applied
            density_of_observable_LAEs_afterEoR_z[z] += LAE_positions_z.density(LAEpos = LAE_EoR)

            #extract luminosities from this model
            #L_model = LAE_positions_z[z].extract_luminosities(LAEpos  =  LAE_EoR)
            #L_model, bins , c = pl.hist( np.log10(L_model.flatten()[L_model.flatten() != 0]), bins = L_bins )

            #make the luminosity a density (the 0.5 is the bin width which needs to be divided)
            #L_model_z[z] = np.true_divide(L_model , float(0.5 * 300 * 300 * 37.5))
    
            #remove the line of sight and luminosity from the list (the acf code doesn't want them - change this in the future)
            rdy_for_acf = LAE_positions.remove_los_from_list(LAEpos = LAE_EoR)
        
            #if there are no LAEs in this slab then there is no point in computing the ACF, skip to the next loop, this is to avoid the xi_2D code throwing an error
            if density_of_observable_LAEs_afterEoR_z[z] < (float(1)/(float(300*300*37.5))):
                os.system("echo the density is 0, we are skipping this slab and moving to the next one")
                continue

            #Compute the ACF
            ACF_model_slab, bins, binn_counter, countbox_list = xi_2D.ps_k(1.5, 20, temperature = rdy_for_acf, mode = 'custom', bins = acfbins)
    
            ACF_model_z[z] += ACF_model_slab
    
    
        #average oibservables over slabs
        density_of_observable_LAEs_afterEoR_z[z] =  density_of_observable_LAEs_afterEoR[z]/float(slabs)
        ACF_model_z = ACF_model_z[z]/float(slabs)
        
        if density_of_observable_LAEs_afterEoR_z[z] < (float(1)/(float(300*300*37.5))):
            os.system("echo the density is 0, perposterous!")
            return -np.inf

    #compute chi squared for each redshift and then add them up
    diff = np.zeros_like(ACF_fiducial_z)
    chi_squared_total = 0
    for z in range(len(z_starts)):
        diff[z] = np.true_divide(ACF_fiducial_z[z] - ACF_model_z[z] , 1)
        chi_squared_total += np.dot(diff[z], diff[z])
    os.system("echo chi squared due to ACF is "  + str(chi_squared_total) )
    chi2_acf.append(chi_squared_total)

    diff = 0
    for z in range(len(z_starts)):
        diff = np.true_divide((density_of_observable_LAEs_afterEoR_z[z] - density_of_observable_LAEs_afterEoR_fiducial_z[z]) , 1)
        chi_squared_total += np.square(diff)
    os.system("echo chi squared due to everything is "  + str(chi_squared_total) )
    chi2_ndensity.append(chi_squared_total)
    
    #print out the results
    #os.system("echo the model and fiducial densities are " + str(density_of_observable_LAEs_afterEoR) + " " + str(density_of_observable_LAEs_afterEoR_fiducial ) + " leading to adding " + str(np.square(np.divide((density_of_observable_LAEs_afterEoR - density_of_observable_LAEs_afterEoR_fiducial), 9e-5 ))) + " chi squared total is "  + str(chi_squared_total) )
    
    #add Luminosity information to the likelihood
    #diff = np.zeros((len(z_starts), len(L_bins))
    #for z in range(len(z_starts)):
    #chi_squared_total += np.dot( np.true_divide( (L_model - L_fiducial) , (L_errors)), np.true_divide( (L_model - L_fiducial) , (L_errors)) ) 

    #cleanup boxes that are leftover
    os.system("rm ../Boxes/*" + str(OUTPUT_NUMBER))
    
    #save the number density of this model
    nmodel_list.append(density_of_observable_LAEs_afterEoR_z)
    acf_list.append(ACF_model_z)

    #announce our results
    os.system("echo RUN " + str(RUN)  + " with params " + str(beta) + " " + str(fduty) + " " + str(zeta) + " " + str(Mturn) + " " +str(M_alpha_min) + " " + str(Rmfp) + " has density " + str(density_of_observable_LAEs_afterEoR))

    #save results to an npz file
    np.savez('MCMC_snapshot' + str(RUN)+ '.npz', betas = np.array(beta_list) , fdutys = np.array(fduty_list), zetas = np.array(zeta_list) , Mturns = np.array(Mturn_list), M_alpha_mins = np.array(M_alpha_min_list), Rmfps = np.array(Rmfp_list) , acf_model = np.array(acf_list) , acf_fiducial = ACF_fiducial, nmodel = density_of_observable_LAEs_afterEoR,  nmock =  np.array(density_of_observable_LAEs_afterEoR_fiducial) , chi2_n = np.array(chi2_ndensity), chi2_ACF = np.array(chi2_acf))
    return -(chi_squared_total)/(2.0)


####################################################################
#                       Make Mock Data                            #
####################################################################
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit()

#parameters used for making fiducial data
#we are using a fiducial inside-out model
fduty_fiducial = 0.025
slabs_fiducial = 8
M_alpha_min_fiducial = 1e10
zeta_fiducial = 500
Mturn_fiducial = 10
Rmfp_fiducial = 30
sign_fiducial = 1
sigma_fiducial = 0.5
intrinsic_params_fiducial = (fduty_fiducial, M_alpha_min_fiducial)
intrinsic_params_fiducial = np.array(intrinsic_params_fiducial)

#Fiducial Boxes:
Fiducial_boxes = '/home/grx40/scratch/21cmFAST_postEoR/Boxes/Fiducial_Boxes/'

#make the fiducial boxes
os.system("./init " + str(sign_fiducial) + ' ' + str(sigma_fiducial) +' ' + str(5015241) )
os.system("./drive_zscroll_noTs " + str(10*zeta_fiducial) +' ' + str(Rmfp_fiducial) +' ' + str(Mturn_fiducial*5*10**7)+ ' '  + str(5015241))

#make fiducial arrays
ACF_fiducial_z = np.zeros((len(z_starts), len(acfbins)))
density_of_observable_LAEs_afterEoR_fiducial_z = np.zeros_like(z_starts)

for z in range(len(z_starts)):
    lightcone_Dictionary_fiducial = {}
    densitylightcone_Dictionary_insideout = {}

    for i in range(slabs*pixelsperslab):
        box_slice = HII_DIM - i -1
        #os.system('echo Making lighrcone for slice ' + str(box_slice))
        lightcone_Dictionary_fiducial[int(box_slice)], lightcone_redshifts_fiducial  = lc.lightcone(DIM = HII_DIM, z_start = z_starts[z], z_end = z_end, N = N, nboxes = nboxes[z], box_slice = int(box_slice),  directory = '../Boxes/', tag = '5015241' , sharp_cutoff = 20, return_redshifts = True )
        densitylightcone_Dictionary[int(box_slice)] = lc.lightcone(DIM = HII_DIM, z_start = z_starts[z], z_end = z_end, N = N, nboxes = nboxes[z], box_slice = int(box_slice), tag = '5015241' ,  directory = '../Boxes/' , marker = 'updated_smoothed_deltax' , sharp_cutoff = 20)

    for sl in range(slabs):
        #these are the observable LAEs without reionization
        Observable_LAEs_fiducial = LAE_positions_z[z].apply_parameters(intrinsic_params_fiducial, LAEpos = LAE_slabs_z[z][str(7 - sl)])

        #compute the density of observable LAEs without a reionization scenario
        density_of_observable_LAEs_fiducial =  LAE_positions_fiducial.density(LAEpos = Observable_LAEs_fiducial)
        
        #apply fiducial reionization scenario (inside out)
        LAE_EoR = LAE_positions.apply_reionization_slabs(lightcone_Dictionary_fiducial, z_range, T_neutral , T_ionized, lightcone_redshifts = lightcone_redshifts_fiducial, density =  densitylightcone_Dictionary,  LAEpos = Observable_LAEs_fiducial)

        #compute the density after reionization model is applied
        density_of_observable_LAEs_afterEoR_fiducial_z[z] +=  LAE_positions_z[z].density(LAEpos = Observable_LAEs_fiducial)

        #remove the line of sight and luminosity from the list (the acf code doesn't want them - change this in the future)
        rdy_for_acf_fiducial = LAE_positions_z[z].remove_los_from_list(LAEpos = LAE_EoR_fiducial)

        #Compute the ACF
        ACF_fiducial, bins_fiducial , binn_counter_fiducial, countbox_list_fiducial = xi_2D.ps_k(1.5, 20, temperature = rdy_for_acf_fiducial, mode = 'custom', bins = acfbins)

        ACF_fiducial_z[z] += ACF_fiducial

        #average oibservables over slabs
        density_of_observable_LAEs_afterEoR_fiducial_z[z] =  density_of_observable_LAEs_afterEoR_fiducial_z[z]/float(slabs)
        ACF_fiducial_z = ACF_fiducial_z[z]/float(slabs)


#save the fiducial number density and angular correlation function
np.savez('fiducial_values_full.npz' , ACF_fiducial_z = ACF_fiducial_z, density_of_observable_LAEs_afterEoR_fiducial_z = density_of_observable_LAEs_afterEoR_fiducial_z)

#remove fiducial boxes
os.system("rm ../Boxes/*" + str(5015241))

####################################################################
#              Define Starting Point and run the MCMC              #
####################################################################
randomize = np.random.normal(1, 0.1, ndim * nwalkers).reshape((nwalkers, ndim))
for i in range(nwalkers):
    randomize[i][0] = np.random.uniform(0.75,0.95)
    randomize[i][1] = np.random.uniform(np.exp(0.5*0.025), np.exp(2*0.025))
    randomize[i][2] = np.random.uniform( 300, 700)
    randomize[i][3] = np.random.uniform(7, 15)
    randomize[i][4] = np.random.uniform(5e9, 8e10)
    randomize[i][5] = np.random.uniform(20, 38)

starting_parameters = np.zeros_like(randomize)
starting_parameters = randomize
#this is only for subsequent runs when we are saving the chains
#npzfile = np.load('checkpoint_values_LAE_full.npz')
#starting_parameters = npzfile['position']
#starting_parameters = np.vstack((starting_parameters, starting_parameters))
os.system('echo Our starting parameters have been saved ')
np.savez('starting_params_full.npz' , starting_parameters = starting_parameters)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool = pool,  args = [ACF_fiducial_z, density_of_observable_LAEs_afterEoR_fiducial_z])
pos , prob , state = sampler.run_mcmc(starting_parameters, 30)


####################################################################
#        Save MCMC results and checkpoint the progress             #
####################################################################

#save final position in an npz file to be ready afterwards
np.savez('checkpoint_values_LAE_full.npz', position = pos, probability = prob, stateof = state, ACFmock = ACF_fiducial_z, nmock =  density_of_observable_LAEs_afterEoR_fiducial_z, acceptance_frac= np.mean(sampler.acceptance_fraction))

#write out chain data to npz files
np.savez('flatchain_LAE_' +str(RUN)+ '_full.npz', betas = sampler.flatchain[:,0],  fduty=sampler.flatchain[:,1], zeta = sampler.flatchain[:,2] ,  Mturn = sampler.flatchain[:,3], M_alpha_min = sampler.flatchain[:,4], Rmfp = sampler.flatchain[:,5] ,  acceptance_frac= np.mean(sampler.acceptance_fraction))


np.savez('chain_LAE_' + str(RUN) +'_full.npz', samples =  sampler.chain)


pool.close()




