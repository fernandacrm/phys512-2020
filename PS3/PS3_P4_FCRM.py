#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:53:37 2020

@author: fernandacristina
"""
import numpy as np
import camb
from matplotlib import pyplot as plt

get_covmat        = 0  # hold tau=0.05 to get the cov matrix
useprevious_pars  = 0  # typed results from above to use to insert_dtau
insert_dtau       = 0  # insert the derivative wrt tau in the cov matrix
useprevious_covmat= 0  # typed results from above to use to get_chain
get_chain         = 0  # float tau to run the chain
fft_chain         = 1
get_chain2        = 0  # it took my computer ~20h to get the first, so not doing the second

pars=np.asarray([65,0.02,0.1,2e-9,0.96])
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

x = wmap[:,0]
y = wmap[:,1]
noise = wmap[:,2]

#run Newton's with numerical derivatives
Ninv = np.eye(len(x))/noise**2
dpar = pars*1e-2
pars = pars
tau  = 0.05
tol  = 1e-6    

# =============================================================================
if get_covmat:
    def get_spectrum(pars,tau=0.05,lmax=2000):
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
        pred  = cmb[2:len(wmap[:,1])+2,0]
        return pred

    def num_deriv(fun,x,pars,dpar):
        derivs = np.zeros([len(x),len(pars)])
        for i in range(len(pars)):
            pars2    = pars.copy()
            pars2[i] = pars2[i]+dpar[i]
            f_right  = fun(pars2)
            pars2[i] = pars[i]-dpar[i]
            f_left   = fun(pars2)
            derivs[:,i] = (f_right-f_left)/(2*dpar[i])
        return derivs
    
    chisq = np.sum( (y-get_spectrum(pars))**2 / noise**2 ) + 2*tol
    
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
        lhs_inv= np.linalg.inv(lhs)
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
    
    par_sigs=np.sqrt(np.diag(lhs_inv))
    data=[x,y,noise]
    # chain,chivec=run_mcmc(pars,data,par_sigs,our_chisq,nstep=10000)
    # par_sigs=np.std(chain,axis=0)
    # par_means=np.mean(chain,axis=0)
    np.savetxt("PS3_P4derivs.txt",derivs)
  
    print()
    print('Fit parameters = ', pars,', with errors = ', par_sigs)
    print('Chi sq = ', chisq)

# =============================================================================
if useprevious_pars:
    pars = np.asarray([6.93273685e+01, 2.24913957e-02, 1.13912236e-01, 2.04250577e-09,
       9.69769787e-01])
    par_sigs = np.asarray([2.40207892e+00, 5.39639860e-04, 5.22680121e-03, 3.89837370e-11,
       1.35890077e-02])
    chisq = 1227.935636135719
    derivs = np.loadtxt('PS3_P4derivs.txt')
    
# =============================================================================
if insert_dtau:
    # guet a derivative for tau and insert it into the covariance matrix
    # (I am not being able to get the cov matrix while floating tau)
    
    new_pars=np.insert(pars,3,tau)
    dtau=tau*1e-2
    
    def get_spectrum(pars,tau=0.05,lmax=2000):
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
        pred  = cmb[2:len(wmap[:,1])+2,0]
        return pred    
    
    # get the function (spectrum) for the two points around tau=0.05
    f_right = get_spectrum(pars,tau+dtau)
    f_left  = get_spectrum(pars,tau-dtau)
    # and the derivative wrt tau
    dfdtau  = (f_right-f_left)/(2*dtau)
    
    # now put this together with the derivatives wrt other pars to get covariance mat
    new_derivs = np.insert(derivs,3,dfdtau,1)
    print(len(new_derivs[0]))
    
    # finally the matrix:
    new_lhs_inv= np.linalg.inv(new_derivs.T @ Ninv @ new_derivs)
    new_errors = np.sqrt(np.diag(new_lhs_inv))
    print('All parameters: ',new_pars,', errors: ',new_errors)

# =============================================================================
if useprevious_covmat:
    new_pars = np.asarray([6.93273685e+01, 2.24913957e-02, 1.13912236e-01, 5.00000000e-02,
       2.04250577e-09, 9.69769787e-01])
    new_lhs_inv = np.asarray([[ 1.34442496e+01,  2.56742125e-03, -2.39773497e-02,
         4.13682441e-01,  1.57862792e-09,  8.47955402e-02],
       [ 2.56742125e-03,  7.22553199e-07, -3.69484024e-06,
         9.80752752e-05,  3.90867500e-13,  2.07927742e-05],
       [-2.39773497e-02, -3.69484024e-06,  4.95016318e-05,
        -7.03316446e-04, -2.60223680e-12, -1.36521523e-04],
       [ 4.13682441e-01,  9.80752752e-05, -7.03316446e-04,
         2.22996118e-02,  8.84439826e-11,  3.34434524e-03],
       [ 1.57862792e-09,  3.90867500e-13, -2.60223680e-12,
         8.84439826e-11,  3.52303330e-19,  1.32140313e-11],
       [ 8.47955402e-02,  2.07927742e-05, -1.36521523e-04,
         3.34434524e-03,  1.32140313e-11,  6.86223452e-04]])
    new_errors = np.asarray([3.66664009e+00, 8.50031293e-04, 7.03573960e-03, 1.49330545e-01,
       5.93551455e-10, 2.61958671e-02])

# =============================================================================
if get_chain:
    def get_spectrum_new(pars,lmax=2000):
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

    def chifun_spectrum(pars,dat):
        x=dat[0]
        y=dat[1]
        errs=dat[2]
        pred=get_spectrum_new(pars)
        chisq=np.sum( ((y-pred)/errs)**2)
        return chisq

    chains  = []
    chivecs = []
    newstep = []
    def run_chain_corr(pars,chifun,data,corr_mat,nsamp=5000,T=1.0):
        npar  = len(pars)
        chain = np.zeros([nsamp,npar])
        chivec= np.zeros(nsamp)
        chisq = chifun(pars,data)
        L     = np.linalg.cholesky(corr_mat)
    
        for i in range(nsamp):
            pars_trial = pars + L @ np.random.randn(npar)*0.4
            if pars_trial[3] > 0:    
                chi_new    = chifun(pars_trial,data)
                delta_chi  = chi_new-chisq
                if np.random.rand(1) < np.exp(-0.5*delta_chi/T):
                    chisq=chi_new
                    pars=pars_trial
                    newstep.append(i) 
            chain[i,:]=pars
            chivec[i]=chisq
            
            chains.append(chain)
            chivecs.append(chivec)
            
        return chain,chivec

    chain,chivec = run_chain_corr(new_pars,chifun_spectrum,[x,y,noise],new_lhs_inv,5000)
   
    np.savetxt("PS3_P4chain.txt",chain)
    np.savetxt("PS3_P4chivec.txt",chivec)
    print('Number of accepted steps:', len(newstep))
    print('Acceptance rate:', len(newstep)/5000)


    pars_chain = np.mean(chain,axis=0)
    errors_chain = np.std(chain,axis=0)
    print('From chain, parameters:',pars_chain,', errors:',errors_chain,
          'chi sq:', chivec[-1])
    
    
    fig, axs = plt.subplots(3,2, figsize=(10,8))#, sharex=True)
    # fig.suptitle('Chains')
    axs[0,0].plot(chain[:,0], color='#2b1d26')
    axs[0,0].legend(['Chain 0'])
    axs[1,0].plot(chain[:,1], color='#4d283f')
    axs[1,0].legend(['Chain 1'])
    axs[2,0].plot(chain[:,2], color='#692a51')
    axs[2,0].legend(['Chain 2'])
    axs[0,1].plot(chain[:,3], color='#852460')
    axs[0,1].legend(['Chain 3'])
    axs[1,1].plot(chain[:,4], color='#bf2685')
    axs[1,1].legend(['Chain 4'])
    axs[2,1].plot(chain[:,5], color='#de6ab2')
    axs[2,1].legend(['Chain 5'])    
    fig.savefig('PS3_P4_figchains.png')
    
    plt.clf();
    plt.plot(chivec, color='#852460')
    plt.legend(['Chivec'])
    plt.savefig('PS3_P4_figchivec.png')

# =============================================================================
if fft_chain:
    chain = np.loadtxt('PS3_P4chain.txt')    

    fft0 = np.abs(np.fft.rfft(chain[:,0]))
    fft1 = np.abs(np.fft.rfft(chain[:,1]))
    fft2 = np.abs(np.fft.rfft(chain[:,2]))
    fft3 = np.abs(np.fft.rfft(chain[:,3]))
    fft4 = np.abs(np.fft.rfft(chain[:,4]))
    fft5 = np.abs(np.fft.rfft(chain[:,5]))

    
    fig, axs = plt.subplots(3,2, figsize=(10,8))#, sharex=True)
    # fig.suptitle('Chains')
    axs[0,0].loglog(fft0[1:], color='#2b1d26')
    axs[0,0].legend(['FFT 0'])
    axs[1,0].loglog(fft1[1:], color='#4d283f')
    axs[1,0].legend(['FFT 1'])
    axs[2,0].loglog(fft2[1:], color='#692a51')
    axs[2,0].legend(['FFT 2'])
    axs[0,1].loglog(fft3[1:], color='#852460')
    axs[0,1].legend(['FFT3 3'])
    axs[1,1].loglog(fft4[1:], color='#bf2685')
    axs[1,1].legend(['FFT 4'])
    axs[2,1].loglog(fft5[1:], color='#de6ab2')
    axs[2,1].legend(['FFT 5'])    
    fig.savefig('PS3_P4_figffts.png')

# =============================================================================
if get_chain2:

    def get_spectrum_new(pars,lmax=2000):
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

    def chifun_spectrum(pars,dat):
        x=dat[0]
        y=dat[1]
        errs=dat[2]
        pred=get_spectrum_new(pars)
        chisq=np.sum( ((y-pred)/errs)**2)
        return chisq

    chains2  = []
    chivecs2 = []
    newstep2 = []
    def run_chain_corr(pars,chifun,data,corr_mat,nsamp=5000,T=1.0):
        npar  = len(pars)
        chain = np.zeros([nsamp,npar])
        chivec= np.zeros(nsamp)
        chisq = chifun(pars,data)
        L     = np.linalg.cholesky(corr_mat)
    
        for i in range(nsamp):
            pars_trial = pars + L @ np.random.randn(npar)*0.4
            if pars_trial[3] > 0:    
                chi_new    = chifun(pars_trial,data)
                delta_chi  = chi_new-chisq
                if np.random.rand(1) < np.exp(-0.5*delta_chi/T):
                    chisq=chi_new
                    pars=pars_trial
                    newstep2.append(i) 
            chain[i,:]=pars
            chivec[i]=chisq
            
            chains2.append(chain)
            chivecs2.append(chivec)
            
        return chain,chivec

    chain = np.loadtxt('PS3_P4chain.txt')    
    pars_guess=np.median(chain,axis=0)
    # step_size_better=np.std(chain[chain.shape[0]//10:,:],axis=0)
    chain2,chivec2=run_chain_corr(pars_guess,chifun_spectrum,[x,y,noise],new_lhs_inv,5000)

# =============================================================================