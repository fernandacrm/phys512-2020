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

"""
Keep tau fixed at 0.05. Use LM minimizer to find best-fit values for 
other pars and errors."""

# The model extendes beyond the region where we have data, so I need to 
# adjust it to get the chisq
# Also need to trim the first two values from the model (data begins at l=2)

import numpy as np
import camb
from matplotlib import pyplot as plt

def get_spectrum(pars,tau=0.05,lmax=2000):
    #print('pars are ',pars)
    H0=pars[0]      # Hubble constant 
    ombh2=pars[1]   # baryon density
    omch2=pars[2]   # cold dark matter density
    tau=tau     # optical depth
    As=pars[3]      # primordial amplitude of fluctuations
    ns=pars[4]      # slope of primordial power law
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    # tt=cmb[:,0]    
    # return tt
    pred  = cmb[2:len(wmap[:,1])+2,0]
    return pred


def num_deriv(fun,x,pars,dpar):
    # calculate numerical derivatives of f wrt its pars
    derivs = np.zeros([len(x),len(pars)])
    for i in range(len(pars)):
        pars2    = pars.copy()
        pars2[i] = pars2[i]+dpar[i]
        f_right  = fun(pars2)
        pars2[i] = pars[i]-dpar[i]
        f_left   = fun(pars2)
        derivs[:,i] = (f_right-f_left)/(2*dpar[i])
    return derivs


pars=np.asarray([65,0.02,0.1,2e-9,0.96])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

x = wmap[:,0]
y = wmap[:,1]
noise = wmap[:,2]

#run Newton's with numerical derivatives
Ninv = np.eye(len(x))/noise**2
# dpar = np.ones(len(pars))*1e-2
dpar = pars*1e-2
pars = pars
tau  = 0.05
tol  = 1e-6     

# cmb   = get_spectrum(pars)
# pred  = cmb[2:len(wmap[:,1])+2]
chisq = np.sum( (wmap[:,1]-get_spectrum(pars))**2 / wmap[:,2]**2 ) + 2*tol

print('chi square = ',chisq)
print('pars = ',pars)
print()


trial =[]
pars_trial =[]
cmb_trial  =[]
deriv_trial=[]
chis  =[]

for i in range(10):
    model = get_spectrum(pars)
    
    chi_trial = np.sum( (y-model)**2 / noise**2 )
    accept_chi = (chisq-chi_trial)
    print('new chi sq = ',chi_trial, ', old - new chi sq = ',accept_chi)
    if 0 < accept_chi < tol: break

    chisq = chi_trial
    derivs= num_deriv(get_spectrum,x,pars,dpar)
    resid = y-model
    lhs   = derivs.T @ Ninv @ derivs
    rhs   = derivs.T @ Ninv @ resid
    lhs_inv= np.linalg.inv(lhs)
    step  = lhs_inv @ rhs
    pars  = pars+step
    print('new pars = ', pars)
    print()
    trial.append(i)
    pars_trial.append(pars)
    cmb_trial.append(get_spectrum(pars))
    deriv_trial.append(derivs)
    chis.append(chi_trial)


par_sigs = np.sqrt(np.diag(lhs_inv))

print()
print('Best fit parameters = ', pars,', with errors = ', par_sigs)
print('Minimized chi sq = ', chisq)

plt.clf()
plt.plot(cmb_trial[0])
plt.plot(deriv_trial[0][:,0])
plt.plot(deriv_trial[0][:,1])
plt.plot(deriv_trial[0][:,2])
# plt.plot(deriv_trial[0][:,3])
plt.plot(deriv_trial[0][:,4])
plt.legend(['Model','deriv par0','deriv par1','deriv par2','deriv par4'])
plt.savefig('PS3_P3_1.png')

plt.clf();
plt.plot(wmap[:,0],wmap[:,1],'.', color='#2b1d26')
plt.plot(cmb_trial[-4])
plt.plot(cmb_trial[-3])
plt.plot(cmb_trial[-2])
plt.plot(cmb_trial[-1], color='#e340ab')
plt.legend(['Data','Trial 1','Trial 2','Trial 3','Best Fit (Trial 4)'])
plt.savefig('PS3_P3_2.png')

plt.clf();
plt.plot(chis,'*', color='#e340ab')
plt.xlabel('Trial');plt.ylabel('$\chi^2$')
plt.savefig('PS3_P3_3.png')

# =============================================================================
# The code for the portion that floats tau is in another file, PS3_P3_2_FCRM.py
