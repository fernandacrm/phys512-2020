#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:57:31 2020

@author: fernandacristina
"""

import numpy as np
import matplotlib.pyplot as plt

def corr_fun(f, g):
    return np.fft.irfft( np.fft.rfft(f) * np.conj(np.fft.rfft(g)) )

def gaussian(x, sigma):
    return np.exp( -0.5*x**2/sigma**2 )

    
x = np.linspace(-50,50,1000)
gauss = gaussian(x,5)
corr_gauss = corr_fun(gauss, gauss)

plt.figure(figsize=(7,2.5))
P1 = plt.plot()
plt.plot(x,gauss,       color='#2b1d26')
plt.plot(x,corr_gauss, color='#e340ab')
plt.legend(['Gaussian','Gaussians correlation'], loc='upper center', fontsize=10)
# plt.savefig('PS4_P2.png',dpi=300)

