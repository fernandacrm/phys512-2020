#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:19:26 2020

@author: fernandacristina
"""

import numpy as np
import matplotlib.pyplot as plt

def shift_fun(f, xshift):
    n = len(f)
    k = np.arange(n)
    return np.fft.ifft( np.fft.fft(f) * np.exp(-2j*np.pi*k*xshift/n) )

def corr_fun(f, g):
    return np.fft.irfft( np.fft.rfft(f) * np.conj(np.fft.rfft(g)) )

def gaussian(x, sigma):
    return np.exp( -0.5*x**2/sigma**2 )

x = np.linspace(-50,50,1000)
gauss = gaussian(x,5)

shifts       = []
shiftgauss   = []
correlations = []

for i in range(10):
    random_shift = np.abs(np.int(np.random.randn()*500))
    shift_gauss  = shift_fun(gauss,random_shift)
    corr_gauss   = corr_fun(gauss, shift_gauss)

    shifts.append(random_shift)
    shiftgauss.append(shift_gauss)
    correlations.append(corr_gauss)

print(shifts)


plt.figure(figsize=(7,3.5))
P1 = plt.plot()
plt.plot(x,gauss, color='#2b1d26', label='Gaussian')
for i in range(10):
    plt.plot(x,correlations[i], label=('Corr shifted by '+np.str(shifts[i]/10)+'%' ) )
    plt.legend(loc='upper left',fontsize=9)
# plt.savefig('PS4_P3.png',dpi=300)

