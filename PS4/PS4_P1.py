#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:11:13 2020

@author: fernandacristina
"""

import numpy as np
import matplotlib.pyplot as plt

def shift_fun(f, xshift):
    n = len(f)
    k = np.arange(n)
    return np.fft.ifft( np.fft.fft(f) * np.exp(-2j*np.pi*k*xshift/n) )

def gaussian(x, sigma):
    return np.exp( -0.5*x**2/sigma**2 )

    
x = np.linspace(-50,50,1000)
gauss = gaussian(x,5)
shift_gauss = shift_fun(gauss,np.int(len(x)/2))

plt.figure(figsize=(7,2.5))
P1 = plt.plot()
plt.plot(x,gauss,       color='#2b1d26')
plt.plot(x,shift_gauss, color='#e340ab')
plt.legend(['Gaussian','Shifted Gaussian'], loc='upper right', fontsize=10)
# plt.savefig('PS4_P1.png',dpi=300)

