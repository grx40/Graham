#!/usr/bin/env python
# coding: utf-8

#This code takes in an arbitrary set of boxes, whose directory and redshift ranges you must provide, and makes a lightcone
import numpy as np
import matplotlib.pyplot as pl
import sys
import os
import astropy
from astropy.cosmology import Planck15 as p15
from astropy.cosmology import z_at_value
from astropy.cosmology import WMAP9 as cosmo


def lightcone(**kwargs ):

    #set defaults:
    mode = 'xH'
    marker = 'xH_'
    N = 500
    DIM = 200
    Box_length = 300
    z_start = 10
    z_end = 6
    nboxes = 21
    directory = '/Users/michael/Documents/MSI-C/21cmFAST/Boxes/z6-10/OriginalTemperatureBoxesNoGaussianities/'
    halo_location_x = 0
    halo_location_y = 0
    slice = DIM - 1

    #sort arguments
    if 'marker' in kwargs:
        marker = kwargs.get('marker')
    if 'DIM' in kwargs:
        DIM = kwargs.get('DIM')
    if 'N' in kwargs:
        N = kwargs.get('N')
    if 'Box_length' in kwargs:
        Box_length = kwargs.get('Box_length')
    if 'z_start' in kwargs:
        z_start = kwargs.get('z_start')
    if 'z_end' in kwargs:
        z_end = kwargs.get('z_end')
    if 'nboxes' in kwargs:
        nboxes = kwargs.get('nboxes')
    if 'directory' in kwargs:
        directory = kwargs.get('directory')
    if 'halo_location_x' in kwargs:
        halo_location_x = kwargs.get('halo_location_x')
    if 'halo_location_y' in kwargs:
        halo_location_y = kwargs.get('halo_location_y')
    if 'box_slice' in kwargs:
        slice = kwargs.get('box_slice')
    if 'return_redshifts' in kwargs:
        return_redshifts = kwargs.get('return_redshifts')
    else:
        return_redshifts = False
    if 'sharp_cutoff' in kwargs:
        sharp_cutoff = kwargs.get('sharp_cutoff')
    else:
        sharp_cutoff = np.inf
    if 'tag' in kwargs:
        tag = str(kwargs.get('tag'))
    else:
        raise ValueError('Supply tag!')


    z_range_of_boxes = np.linspace(z_start,z_end,nboxes)
#print(z_end)
#   print(z_start)
#   print(nboxes)
#print(z_range_of_boxes)

    ####################################################
    # useful functions
    ####################################################

    def box_maker(name):       #reads in the boxes
        data = np.fromfile(directory + name ,dtype=np.float32)
        box = data.reshape((DIM,DIM,DIM))
        return box

    box_path = np.zeros((len(z_range_of_boxes)), dtype = 'object')
    box_path_redshifts = np.zeros((len(box_path)))
    for fn in os.listdir(directory):   #parse the boxes
        #print(fn)
        #we only need 2 boxes in practice
        for z in range(nboxes):
            #print('checking for marker' , fn) 
            if marker in fn[0:25] and str(np.round(z_range_of_boxes[z],2)) in fn[0:32] and tag in fn:
                #print(z_range_of_boxes[z], fn)
                #print(marker, fn[0:25], fn[0:32])
                box_path[z] = fn
                index = box_path[z].find('_z0')
                start = index + len('_z0')
                box_path_redshifts[z] = float(box_path[z][start:start + 4])
                if box_path_redshifts[0] == box_path_redshifts[1]:
                    os.system("echo ruh roh this is fucked " + str(box_path[z]) + " and redshift " + str(box_path_redshifts[z]) + " tag is " + str(tag) + " " + str(np.round(z_range_of_boxes[z],2)) + " " + str(fn) )
                    os.system("echo searching for z " + str(z) + " within filename "  + str(fn) + " for redshift " + str(np.round(z_range_of_boxes[z],2)) + " box_path history is " + str(box_path[0]) + "  " + str(box_path[1]))
                    os.system("echo the marker was " + str(marker))
                    os.system("echo list of directory will now be saved")
                    os.system("echo index was " + str(index) + " start was " + str(start) )
                    #np.savez('Debug.npz' , fns = os.listdir(directory), box_path_redshifts = box_path_redshifts, box_path = box_path, z_range_of_boxes = z_range_of_boxes)

#print(box_path_redshifts)
#   print(box_path)

    #function to determine weighting of each redshift to boxes
    def find_sandwiched_bins(z_range, z):
        z_floor = np.max(z_range)
        z_ceil = z_range[1]
        binn = 1
        while(z_ceil >= np.min(z_range)):
            if ((z <= z_floor) and (z > z_ceil)):
                return ( z_ceil, z_floor)
            z_floor = z_ceil
            if z_ceil == np.max(z_range):
                break
            z_ceil = z_range[binn+1]
            binn += 1
            #safety net
            if binn > 1000:
                os.system("echo breaking")
                break


    def comoving2pixel(DIM, Box_length, comoving_distance):
        return int(float(comoving_distance * DIM)/float(Box_length))

    def didweswitchbox(historyofzminus, z_plus, ctr):
        if z_plus < historyofzminus[ctr - 1 ]:
            return True
        else:
            return False

    ####################################################
    # initialize all relevant arrays
    ####################################################

    lightcone = np.zeros((N, DIM, DIM))
    lightcone_halo = np.zeros((N))
    z_range = np.linspace(z_start,z_end,N)
    zs = []
    z = z_range[0]
    ctr = 0
    comoving_distance_z0_zstart = cosmo.comoving_distance(z_range[0]).value
    prev_pix_loc = 0
    pixel_addition = 0
    pixel_origin = 0
    pixel_location_relative_to_origin = 0
    historyofzminus = []

    ####################################################
    # loop through redshifts
    ####################################################
    while(z > np.min(z_range)):

        #comment out the bottom few lines because we are only using two boxes, so the sandwiched bins are always going to be the same
        z_sandwhich = find_sandwiched_bins(box_path_redshifts, z)
        
        if type(z_sandwhich[0]) ==  type(None) or type(z_sandwhich[1]) ==  type(None):
            os.system("echo looking for redshift , output number" + str(z) + " " + str(tag))
            os.system("echo box path redshifts" + str(box_path_redshifts))
            os.system("echo z_sandwhich is " + str(z_sandwhich))
            np.savez('Debug2.npz' , fns = os.listdir(directory), box_path_redshifts = box_path_redshifts, box_path = box_path)

        z_minus = z_sandwhich[0]
        z_plus = z_sandwhich[1]

        historyofzminus.append(z_plus)
        xH_minus = box_maker(box_path[list(box_path_redshifts).index(z_minus)])
        xH_plus = box_maker(box_path[list(box_path_redshifts).index(z_plus)])
        
        comoving_distance_z = cosmo.comoving_distance(z).value
        comoving_distance_z0_to_z = comoving_distance_z0_zstart - comoving_distance_z
        comoving_distance_from_last_switch = cosmo.comoving_distance(z_plus).value
        
        
        if ctr == 0:
            pixel_addition = comoving2pixel(DIM,Box_length, comoving_distance_z0_to_z)
            prev_pix_loc = -pixel_addition + slice
            pixel_origin = slice

        else:
            if didweswitchbox(historyofzminus, z_plus, ctr):
                #print(z_plus, z , historyofzminus[ctr-1])
                pixel_origin = prev_pix_loc
            pixel_location_relative_to_origin = -comoving2pixel(DIM,Box_length, comoving_distance_from_last_switch - comoving_distance_z)
            pixel_addition = (pixel_location_relative_to_origin + pixel_origin)%DIM
            prev_pix_loc = pixel_addition

        zs.append(z)
        
        lightcone[ctr,:,:] =  (xH_plus[pixel_addition,:,:] - xH_minus[pixel_addition,:,:])*((z - z_minus)/(z_plus - z_minus)) + xH_minus[pixel_addition,:,:]
    
        lightcone_halo[ctr] =  (xH_plus[pixel_addition][halo_location_x][halo_location_y] - xH_minus[pixel_addition][halo_location_x][halo_location_y])*((z - z_minus)/(z_plus - z_minus)) + xH_minus[pixel_addition][halo_location_x][halo_location_y]

        ctr += 1
        z = z_range[ctr]
        #pl.savefig(str(ctr)+'.png')
        #safety net
        if ctr > N:
            break
        if ctr >= sharp_cutoff:
            if return_redshifts:
                return lightcone[1:sharp_cutoff,:,] , np.array(zs[1:])
            else:
                return lightcone[1:sharp_cutoff,:,]

    #return the lightcone history as an array across all redshifts
    
    if return_redshifts:
        return lightcone , np.array(zs)
    else:
        return lightcone


#lightconepng = lightcone(N = 500 )
#directory = '/Users/michael/Research/LAE_Clustering/Boxes_w_HaloFinder/'

#pl.imshow(np.swapaxes(lightconepng,0,2)[100])
#pl.savefig('Lightcone.png')
#pl.ylabel('Box slice at x = 0')
#pl.xlabel('Redshift')
#pl.show()
#pl.close()


