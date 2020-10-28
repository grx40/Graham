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
nwalkers = 96



####################################################################
#                     LAE script parameters               #
####################################################################
#Constants for the script
slabs = 8
pixelsperslab = 25
Mpcperslab = 37.5
Box_length = 300
lya_min = 2.5e42
HII_DIM = 200
DIM = 800

z_start = 6.6
z_end = 6.4
N = 150
nboxes = 2

z_range = np.linspace(z_start, z_end, nboxes)
box = 'halos_z6.60_800_300Mpc_82600201'

#Load the halo file that we will be using as a template (This is possiby the master halo list). Load this only once for the entire run
halopos = np.genfromtxt( box, dtype=None)
print('the shape of the halo list' , halopos.shape)
#acfbins = np.linspace(0.5, 300, 100)
#below are the acf bins which match the ouchi data
#acfbins = (0.5, 2, 7, 10,  24.5, 50)
acfbins = xi_2D.create_k_boundaries(0.008, 1.3 , mode = 'custom', bins = np.linspace(1,50, 50))[1]

#below are the theoretical bins used for the Mathee et all luminosity function
L_fiducial = (1e-3, 2.13e-4, 8.1e-6)
L_bins = (42.5, 43, 43.5, 44)
L_errors = (3e-4, 1e-4, 1e-5)

ACF_fiducial = (0.6228, 0.20785, 0.020785)
ACF_error = (0.4, 0.12, 0.02)

####################################################################
#                 Define Bayesian Probabilities                    #
####################################################################

#lets add the number density to the prior. If we constrain it using the likelihood then we may end up in the
#unfortunate situation of having the code get stuck with a gigantic ACF
def lnprior(x,  density_of_observable_LAEs_afterEoR_fiducial):
    beta = x[0]
    fduty = x[1]
    zeta = x[2]
    Mturn = x[3] 
    M_alpha_min = x[4]
    Rmfp = x[5]

    n_density  = density_of_observable_LAEs_afterEoR_fiducial
    if  -1  < beta < 1 and 0 < np.exp(fduty) <= 1 and  200 < zeta < 1000 and 1e7 < (Mturn*5e7) < 9e9 and 1e9 < M_alpha_min < 1e11 and 5 < Rmfp < 60:
        os.system("echo RUN " + str(RUN) + " accepting the fuck out of beta and fduty " + str(beta) + " " + str(np.exp(fduty)) + " " + str(zeta) + " " + str(Mturn) +" " + str(M_alpha_min/float(1e10)) + " " + str(Rmfp)   )
        return 0.0
    os.system("echo RUN " + str(RUN) + " Rejecting the fuck out of beta and fduty " + str(n_density) + " " + str(beta) + " " + str(np.exp(fduty)) + " " + str(zeta) + " " + str(Mturn) +" " + str(M_alpha_min/float(1e10)) + " " + str(Rmfp ) )
    return -np.inf


def lnprob(x, ACF_fiducial, density_of_observable_LAEs_afterEoR_fiducial, L_fiducial ):
    lp = lnprior(x, density_of_observable_LAEs_afterEoR_fiducial)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(x, ACF_fiducial, density_of_observable_LAEs_afterEoR_fiducial, L_fiducial)

beta_list = []
fduty_list = []
zeta_list = []
Mturn_list = []
M_alpha_min_list = []
Rmfp_list = []
acf_list = []
def lnlike(x, ACF_fiducial, density_of_observable_LAEs_afterEoR_fiducial, L_fiducial):
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
    #os.system("echo beta and fduty are " +  str(beta) + " " + str(fduty))
    
    t21_i = time.time()
    #make the reionization scenario for these parameters
    os.system("echo choice of beta is "  + str(beta) + ' leading to a sigma of' + str(sigma) +' with sign' + str(sign) )
    os.system("./init " + str(sign) + ' ' + str(sigma) +' ' + str(OUTPUT_NUMBER) )
    os.system("./drive_zscroll_noTs " + str(10*zeta) +' ' + str(Rmfp) +' ' + str(Mturn*5*10**7)+ ' '  + str(OUTPUT_NUMBER))
    t21_f = time.time()
    os.system("echo 21cmfast runtime is " + str(t21_f - t21_i))

    #let us make the lightcone for this reionization scenario
    lightcone_Dictionary = {}
    density_lightcone_Dictionary = {}
    start = time.time()
    ctr = 1
    for i in range(pixelsperslab):
        box_slice = HII_DIM - i -1
        os.system('echo Making lighrcone for slice ' + str(box_slice))
        lightcone_Dictionary[int(box_slice)] , lightcone_redshifts = lc.lightcone(DIM = HII_DIM, z_start = z_start, z_end = z_end, N = N, nboxes = nboxes, box_slice = int(box_slice), directory = '../Boxes/', sharp_cutoff = 20, return_redshifts = True , tag = str(OUTPUT_NUMBER) )
        density_lightcone_Dictionary[int(box_slice)], lightcone_redshifts  = lc.lightcone(DIM = HII_DIM, z_start = z_start, z_end = z_end, N = N, nboxes = nboxes, box_slice = int(box_slice), directory = '../Boxes/', marker = 'updated_smoothed' ,  sharp_cutoff = 20, return_redshifts = True , tag = str(OUTPUT_NUMBER) )
        ctr += 1
    end = time.time()
    os.system("RUNTIME OF LIGHTCONE (s) : " + str( end - start))

    #okay, we are ready to apply the scenario corresponding to these parameters onto the master list of halos

    #make an instance of this class with the loaded haloposition from 21cmFAST
    LAE_positions = LAEC.LAE_Cluster(halopos, HII_DIM, DIM, Box_length)
    #sort the list into slabs
    LAE_slabs = LAE_positions.sort_into_slabs(slabs = 8, pixelsperslab = 25)

    #intrinsic luminosity params for this model are
    intrinsic_params = (fduty, M_alpha_min)

    #these are the observable LAEs without reionization
    Observable_LAEs = LAE_positions.apply_parameters(intrinsic_params, LAEpos = LAE_slabs['7'] )

    #compute the density of observable LAEs without a reionization scenario
    density_of_observable_LAEs =  LAE_positions.density(LAEpos = Observable_LAEs)

    #apply fiducial reionization scenario (inside out)
    LAE_EoR = LAE_positions.apply_reionization_slabs(lightcone_Dictionary, z_range, T_neutral , T_ionized, lightcone_redshifts = lightcone_redshifts, density =  density_lightcone_Dictionary,  LAEpos = Observable_LAEs)

    #compute the density after reionization model is applied
    density_of_observable_LAEs_afterEoR =  LAE_positions.density(LAEpos = LAE_EoR)
    
    if density_of_observable_LAEs_afterEoR < (float(1)/(float(300*300*37.5))):
        os.system("echo the density is 0, we are checking out of this loop")
        return -np.inf

    #extract luminosities from this model
    L_model = LAE_positions.extract_luminosities(LAEpos  =  LAE_EoR)
    L_model, bins , c = pl.hist( np.log10(L_model.flatten()[L_model.flatten() != 0]), bins = L_bins )

    #make the luminosity a density (the 0.5 is the bin width which needs to be divided)
    L_model = np.true_divide(L_model , float(0.5 * 300 * 300 * 37.5))
    os.system("echo RUN " + str(RUN)  + " with params " + str(beta) + " " + str(fduty) + " " + str(zeta) + " " + str(Mturn) + " " +str(M_alpha_min) + " " + str(Rmfp) + " has density " + str(density_of_observable_LAEs_afterEoR))
    #remove the line of sight and luminosity from the list (the acf code doesn't want them - change this in the future)
    rdy_for_acf = LAE_positions.remove_los_from_list(LAEpos = LAE_EoR)
    
    #os.system("echo --------------Below are the results for the model parameters ---------------------")
    os.system("echo Before applying a reionization scenario, there are a total average density of " + str(density_of_observable_LAEs ) )
    os.system("echo After applying the reionization scenario, there are a total average density of" + str(density_of_observable_LAEs_afterEoR) )

    #Compute the ACF
    ACF_model, bins, binn_counter, countbox_list = xi_2D.ps_k(1.5, 20, temperature = rdy_for_acf, mode = 'custom', bins = acfbins)
    acf_list.append(ACF_model)
    #os.system('echo model params ' + str(fduty) + ' ' + str(beta) + ' is ' + str(zeta*0.05) + ' ' + str(Mturn) + ' ' + str(M_alpha_min) + ' ' + str(Rmfp) )

    diff = np.zeros_like(ACF_fiducial)
    #compute chi squared for each redshift and then add them up

    #subsample
    #ACF_model_subsample = (ACF_model[1] , ACF_model[2] , ACF_model[4])
    ACF_model_subsample = (ACF_model[1] , ACF_model[6] , ACF_model[24])
    ACF_model_subsample = np.array(ACF_model_subsample)

    chi_squared_total = 0
    for i in range(len(ACF_fiducial)):
        #this makes an array of diff[redshift][bin]
        #previously error was 10**-4
        diff[i] = np.divide((ACF_model_subsample[i] - ACF_fiducial[i]), ACF_error[i] )
        chi_squared_total += np.dot(diff[i], diff[i])
        os.system("echo chi squared is "  + str(chi_squared_total) )

    #add density to the likelihood
    os.system("echo the model and fiducial densities are " + str(density_of_observable_LAEs_afterEoR) + " " + str(density_of_observable_LAEs_afterEoR_fiducial ) )
    chi_squared_total += np.square(np.true_divide((density_of_observable_LAEs_afterEoR - density_of_observable_LAEs_afterEoR_fiducial), 5e-5 ))
   
    #add Luminosity information to the likelihood
    #chi_squared_total += np.dot( np.true_divide( (L_model - L_fiducial) , (L_errors)), np.true_divide( (L_model - L_fiducial) , (L_errors)) ) 

    os.system("echo leading to adding " + str(np.square(np.divide((density_of_observable_LAEs_afterEoR - density_of_observable_LAEs_afterEoR_fiducial), 5e-5 ))))  
    os.system("echo chi squared total is "  + str(chi_squared_total))

    #cleanup boxes that are leftover
    os.system("rm ../Boxes/*" + str(OUTPUT_NUMBER))

    #save results to an npz file
    np.savez('MCMC_snapshot' + str(RUN)+ '.npz', betas = np.array(beta_list) , fdutys = np.array(fduty_list), zetas = np.array(zeta_list) , Mturns = np.array(Mturn_list), M_alpha_mins = np.array(M_alpha_min_list), Rmfps = np.array(Rmfp_list) , acf_model = np.array(acf_list) , acf_fiducial = ACF_fiducial, nmock =  density_of_observable_LAEs_afterEoR_fiducial)
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
intrinsic_params_fiducial = (fduty_fiducial, M_alpha_min_fiducial)
intrinsic_params_fiducial = np.array(intrinsic_params_fiducial)

#Fiducial Boxes:
Fiducial_boxes = '/home/grx40/scratch/21cmFAST_postEoR/Boxes/Fiducial_Boxes/'

#start = time.time()
#lightcone_Dictionary_fiducial = {}
#densitylightcone_Dictionary_insideout = {}
#ctr = 1
#for i in range(pixelsperslab):
#    box_slice = HII_DIM - i -1
#    os.system('echo Making lighrcone for slice ' + str(box_slice))
#    lightcone_Dictionary_fiducial[int(box_slice)], lightcone_redshifts_io  = lc.lightcone(DIM = HII_DIM, z_start = z_start, z_end = z_end, N = N, nboxes = nboxes, box_slice = int(box_slice), directory = Fiducial_boxes, tag = '82600201' , sharp_cutoff = 20, return_redshifts = True )
#    densitylightcone_Dictionary_insideout[int(box_slice)] = lc.lightcone(DIM = HII_DIM, z_start = z_start, z_end = z_end, N = N, nboxes = nboxes, box_slice = int(box_slice), tag = '82600201' ,  directory = Fiducial_boxes , marker = 'updated_smoothed_deltax' , sharp_cutoff = 20)
#    ctr += 1
#end = time.time()
#os.system("RUNTIME (s) : " + str( end - start))


#make an instance of this class with the loaded haloposition from 21cmFAST
#LAE_positions_fiducial = LAEC.LAE_Cluster(halopos, HII_DIM, DIM, Box_length)
#sort the list into slabs
#LAE_slabs_fiducial = LAE_positions_fiducial.sort_into_slabs(slabs = 8, pixelsperslab = 25)

#these are the observable LAEs without reionization
#Observable_LAEs_fiducial = LAE_positions_fiducial.apply_parameters(intrinsic_params_fiducial, LAEpos = LAE_slabs_fiducial['7'] )

#compute the density of observable LAEs without a reionization scenario
#density_of_observable_LAEs_fiducial =  LAE_positions_fiducial.density(LAEpos = Observable_LAEs_fiducial)
os.system("echo lets check to make sure we're computing densities correctly, the following should match")
#os.system("echo density 1 " + str(density_of_observable_LAEs_fiducial))
#Box_Count, Box_Luminosity  = LAE_positions_fiducial.map_slab2box(LAEpos = Observable_LAEs_fiducial)
#os.system("echo density 2" + str( float(np.sum(Box_Count))/float(37.5*300*300 )))
#apply fiducial reionization scenario (inside out)
#LAE_EoR_fiducial = LAE_positions_fiducial.apply_reionization_slabs(lightcone_Dictionary_fiducial , z_range, T_neutral, T_ionized, lightcone_redshifts = lightcone_redshifts_io, density =  densitylightcone_Dictionary_insideout,  LAEpos = Observable_LAEs_fiducial )

#compute the density after reionization model is applied
#density_of_observable_LAEs_afterEoR_fiducial =  LAE_positions_fiducial.density(LAEpos = Observable_LAEs_fiducial)

#remove the line of sight and luminosity from the list (the acf code doesn't want them - change this in the future)
#rdy_for_acf_fiducial = LAE_positions_fiducial.remove_los_from_list(LAEpos = LAE_EoR_fiducial)


#os.system("echo --------------Below are the results for the fiducial---------------------")
#os.system("echo Before applying a reionization scenario, there are a total average density of " + str(density_of_observable_LAEs_fiducial ) )
#os.system("echo After applying the fiducial reionization scenario, there are a total average density of" + str(density_of_observable_LAEs_afterEoR_fiducial) )

#Compute the ACF
#ACF_fiducial, bins_fiducial , binn_counter_fiducial, countbox_list_fiducial = xi_2D.ps_k(1.5, 20, temperature = rdy_for_acf_fiducial, mode = 'custom', bins = acfbins)

#os.system('echo ACF is '+ str(ACF_fiducial))
#os.system("rm ../Boxes/*" + str(OUTPUT_NUMBER_MOCK))

####################################################################
#              Define Starting Point and run the MCMC              #
####################################################################
#beta0 , fduty0 , zeta0, Mturn, M_alpha_min0 , Rmfp0 = 0.5 , 0.02, 400, 5e7, 1e10, 30
#randomize = np.random.normal(1, 0.1, ndim * nwalkers).reshape((nwalkers, ndim))
#starting_parameters = randomize*np.array([[beta0, fduty0, zeta0, Mturn, M_alpha_min0, Rmfp0 ]]*nwalkers)
#for i in range(nwalkers):
#    randomize[i][0] = np.random.uniform(0.25,0.95)
#    randomize[i][1] = np.random.uniform(np.log(0.01), np.log(0.1))
#    randomize[i][2] = np.random.uniform( 300, 700)
#    randomize[i][3] = np.random.uniform(5, 15)
#    randomize[i][4] = np.random.uniform(5e9, 8e10)
#    randomize[i][5] = np.random.uniform(20, 38)

#starting_parameters = randomize

npzfile = np.load('checkpoint_values_LAE_full.npz')
starting_parameters = npzfile['position']
#starting_parameters = np.vstack((starting_parameters, starting_parameters))


#npzfile = np.load('checkpoint_values_LAE_full.npz')
#starting_parameters = npzfile['position']
#starting_parameters = np.vstack((starting_parameters, starting_parameters))


os.system('echo Our starting parameters have been saved ')
np.savez('starting_params_full.npz' , starting_parameters = starting_parameters)


density_of_observable_LAEs_afterEoR_fiducial = 4.1e-4

#L_fiducial for all bins, we are only using a subset of these
L_fiducial = (1e-3, 2.13e-4, 8.1e-6)
L_bins = (42.5, 43, 43.5, 44)
L_errors = (3e-4, 1e-4, 1e-5)

ACF_fiducial = (0.6228, 0.20785, 0.020785)
ACF_error = (0.4, 0.12, 0.02)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool = pool,  args = [ACF_fiducial, density_of_observable_LAEs_afterEoR_fiducial, L_fiducial])
#burn in
#pos , prob , state = sampler.run_mcmc(starting_parameters, 1)
#sampler.reset()
pos , prob , state = sampler.run_mcmc(starting_parameters, 30)


####################################################################
#        Save MCMC results and checkpoint the progress             #
####################################################################

#save final position in an npz file to be ready afterwards
np.savez('checkpoint_values_LAE_full.npz', position = pos, probability = prob, stateof = state, ACFmock = ACF_fiducial, nmock =  density_of_observable_LAEs_afterEoR_fiducial, acceptance_frac= np.mean(sampler.acceptance_fraction))

#write out chain data to npz files
np.savez('flatchain_LAE_' +str(RUN)+ '_full.npz', betas = sampler.flatchain[:,0],  fduty=sampler.flatchain[:,1], zeta = sampler.flatchain[:,2] ,  Mturn = sampler.flatchain[:,3], M_alpha_min = sampler.flatchain[:,4], Rmfp = sampler.flatchain[:,5] ,  acceptance_frac= np.mean(sampler.acceptance_fraction))


np.savez('chain_LAE_' + str(RUN) +'_full.npz', samples =  sampler.chain)


pool.close()




