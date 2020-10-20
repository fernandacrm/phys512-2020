#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:11:16 2020

@author: fernandacristina
"""
""""
Use the power spectrum of the Cosmic Microwave Background (CMB) to constrain
basic cosmological parameters of the universe.
Parameters to be measured: 
    Hubble constant, 
    density of regular baryonic matter, 
    density of dark matter, 
    amplitude and tilt of initial power spectrum of fluctuations in very early universe,
    Thomson scattering optical depth between us and CMB.
"""

import numpy as np
import camb
from matplotlib import pyplot as plt
import time

def get_spectrum(pars,lmax=2000):
    #print('pars are ',pars)
    H0=pars[0]      # Hubble constant 
    ombh2=pars[1]   # baryon density
    omch2=pars[2]   # cold dark matter density
    tau=pars[3]     # optical depth
    As=pars[4]      # primordial amplitude of fluctuations
    ns=pars[5]      # slope of primordial power law
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    
    return tt

pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

cmb=get_spectrum(pars)

plt.clf();
#plt.errorbar(wmap[:,0],wmap[:,1],wmap[:,2],fmt='*')
plt.plot(wmap[:,0],wmap[:,1],'.', color='#2b1d26')
plt.plot(cmb, color='#e340ab')
plt.legend(['Data','Fit'])
plt.savefig('PS3_P2.png')

# The model extendes beyond the region where we have data, so I need to 
# adjust it to get the chisq
# Also need to trim thw first two values from the model

pred  = cmb[2:len(wmap[:,1])+2]
chisq = np.sum( (wmap[:,1]-pred)**2 / wmap[:,2]**2 )
print('chi square = ',chisq)
