#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:17:10 2020

@author: fernandacristina
"""
import numpy as np
import camb
from matplotlib import pyplot as plt

float_tau=1
if float_tau:
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


    pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
    wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

    x = wmap[:,0]
    y = wmap[:,1]
    noise = wmap[:,2]

    #run Newton's with numerical derivatives
    Ninv = np.eye(len(x))/noise**2
    # dpar = np.ones(len(pars))*1e-2
    dpar = pars*1e-2
    pars = pars
    tol  = 1e-6     

    chisq = np.sum( (wmap[:,1]-get_spectrum(pars))**2 / wmap[:,2]**2 ) + 2*tol

    print('chi square = ',chisq)
    print('pars = ',pars)
    print()

    trial          = []
    model_trials   = []
    chi_trials     = []
    deriv_trials   = []
    resid_trials   = []
    lhs_trials     = []
    lhs_inv_trials = []
    step_trials    = []
    pars_trials    = []

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
        lhs_inv= np.linalg.pinv(lhs)
        step  = lhs_inv @ rhs
        pars  = pars+step
        print('new pars = ', pars)
        print()
        trial.append(i)
        model_trials.append(model)
        chi_trials.append(chi_trial)
        deriv_trials.append(derivs)
        resid_trials.append(resid)
        lhs_trials.append(lhs)
        lhs_inv_trials.append(lhs_inv)
        step_trials.append(step)
        pars_trials.append(pars)

    pars_trials_reverse    = pars_trials[::-1]
    lhs_inv_trials_reverse = lhs_inv_trials[::-1]
    chi_trials_reverse     = chi_trials[::-1]
    model_trials_reverse   = model_trials[::-1]

    for i in range(len(pars_trials_reverse)):
        check_tau=pars_trials_reverse[i][3]
        print(i,check_tau)
        if check_tau > 0:
            use_pars = pars_trials_reverse[i]
            use_errors = lhs_inv_trials_reverse[i]
            par_errors = np.sqrt(np.diag(use_errors))
            use_model  = model_trials_reverse[i]
            print(i,'Parameters to use: ',use_pars)
            print('Errors: ',par_errors)
            print('Chi sqr: ',chi_trials_reverse[i])
            break
            
plt.clf();
plt.plot(wmap[:,0],wmap[:,1],'.', color='#2b1d26')
plt.plot(use_model, color='#e340ab')
plt.legend(['Data','Model (float all)'])
plt.savefig('PS3_P3_4.png')
