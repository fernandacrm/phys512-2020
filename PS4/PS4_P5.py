#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:48:14 2020

@author: fernandacristina
"""

import numpy as np
import matplotlib.pyplot as plt

saving = 0

def myfft(N,k):
    x    = np.arange(N)
    kvec = np.arange(N)
    FT = []
    for K in kvec:
        FTK = np.sum(1/(2J)*(np.exp(-2J*np.pi*(K-k)*x/N)-np.exp(-2J*np.pi*(K+k)*x/N)))
        FT.append(np.abs(FTK))
    return kvec, FT

n = 256
x = np.arange(n)

ks     = []; fts     = []
k_ints = []; ft_ints = []

for i in range(10):
    k = np.abs(np.random.randn()*100)
    ks.append(k)
    fts.append(myfft(n,k))
    
    k_int = np.int(k)
    k_ints.append(k_int)
    ft_ints.append(myfft(n,k_int))
    


plt.figure(figsize=(7,2.5))
plt.plot()
for i in range(5):
    plt.plot(fts[i][0],fts[i][1], label='k = '+np.str('%.2f' % ks[i]))
    plt.legend( loc='upper right', fontsize=9)
plt.xlabel("k'")
plt.tight_layout()
if saving:
    plt.savefig('PS4_P5cnonint2.png',dpi=300)
    

plt.figure(figsize=(7,2.5))
plt.plot()
for i in range(5):
    plt.plot(ft_ints[i][0],ft_ints[i][1], label='k = '+np.str('%.2f' % k_ints[i]))
    plt.legend( loc='upper right', fontsize=9)
plt.xlabel("k'")
plt.tight_layout()
if saving:
    plt.savefig('PS4_P5cint2.png',dpi=300)



y = np.sin(2 * np.pi *x*ks[0]/n)
w = 0.5 - 0.5*np.cos(2 * np.pi *x/n) 
ftnp       = np.fft.fft(y)
ftnpwindow = np.fft.fft(y*w)

ftw = np.fft.fft(w)

plt.figure(figsize=(7,2.5))
plt.plot()
plt.plot(np.abs(ftnp), label='No window, k = '+np.str('%.2f' % ks[0]),color='#2b1d26')
plt.plot(np.abs(ftnpwindow), label='Window, k = '+np.str('%.2f' % ks[0]), color='#e340ab')
plt.legend( loc='upper center', fontsize=9)
plt.xlabel("k'")
plt.tight_layout()
if saving:
    plt.savefig('PS4_P5d.png',dpi=300)

    
plt.figure(figsize=(7,2.5))
plt.plot()
plt.plot(ftw, label='FFT{window} (N=256)',color='#e340ab')
plt.legend( loc='upper center', fontsize=9)
plt.xlabel("k'")
plt.yticks([-n/4,0.,n/4,n/2])
plt.tight_layout()
if saving:
    plt.savefig('PS4_P5e.png',dpi=300)
