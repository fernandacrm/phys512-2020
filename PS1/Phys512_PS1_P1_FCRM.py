#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 02:07:50 2020

@author: fernandacristina
"""

"""
PHYS 512 - PS1 - P1

I am evaluating the function at x0=1 with the derivatives
from x+/-delta and x+/-2delta

Want to check the best step size delta (or the order of magnitude of delta
that gives the best accuracy)

So I will compare the results from a calculated derivative with different 
choices of delta with the true analytical derivative.
"""
  
# =============================================================================
'Derivatives from x+/-delta and x+/-2delta for f(x)=exp(x)'

import numpy as np

orders=np.linspace(-20,-1,77)    #potential orders of magnitude to try for delta
x0=1    #point where I am evaluating f(x)

exp_x=1
if exp_x:
    
    f0=np.exp(x0)
    deriv_true=np.exp(x0)
    
    deltas=[]
    errors=[]
    
    for order in orders:
        delta  = 10**order
        f1plus = np.exp(x0+delta)
        f1minus= np.exp(x0-delta)
        
        f2plus = np.exp(x0+2*delta)
        f2minus= np.exp(x0-2*delta)
        
        deriv = (2/3*f1plus-2/3*f1minus-1/12*f2plus+1/12*f2minus)/(delta)
        
        print(order,delta,np.abs(deriv-deriv_true))
        
        deltas.append(10**order)
        errors.append(np.abs(deriv-deriv_true))
        
    besterror    =np.amin(errors)
    bestdelta_arg=np.argmin(errors)
    bestdelta    =deltas[bestdelta_arg]


    print('For f(x)=exp(x), Best delta = ',bestdelta,', best accuracy = ',besterror)

  
# =============================================================================
'Derivatives from x+/-delta and x+/-2delta for f(x)=exp(.01x)'

exp_01x=1
if exp_01x:
    
    f0=np.exp(.01*x0)
    deriv_true=.01*np.exp(.01*x0)

    deltas=[]
    errors=[]
    
    for order in orders:
        delta  = 10**order
        f1plus = np.exp(.01*(x0+delta))
        f1minus= np.exp(.01*(x0-delta))
        
        f2plus = np.exp(.01*(x0+2*delta))
        f2minus= np.exp(.01*(x0-2*delta))    
        
        deriv = (2/3*f1plus-2/3*f1minus-1/12*f2plus+1/12*f2minus)/(delta)

        # print(order,delta,np.abs(deriv-deriv_true))
        
        deltas.append(10**order)
        errors.append(np.abs(deriv-deriv_true))
        
        besterror   =np.amin(errors)
        bestdelta_arg=np.argmin(errors)
        bestdelta=deltas[bestdelta_arg]

                
    print('For f(x)=exp(.01*x), Best delta = ',bestdelta,', best accuracy = ',besterror)
    