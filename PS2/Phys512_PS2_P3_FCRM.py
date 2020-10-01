#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 00:26:53 2020

@author: fernandacristina
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import time

# =============================================================================
"""Conversions to seconds"""
mins = 60
hs   = 60*mins
days = 24*hs
ys   = 365*days 

elements = ['U238', 'Th234', 'Pa234', 'U234', 'Th230',
            'Ra226', 'Rn222', 'Po218', 'Pb214', 'Bi210',
            'Po214', 'Pb210', 'Bi210', 'Po210', 'Pb206']

    
hl=[4.468e9*ys, 24.10*days, 6.70*hs, 245500*ys, 75380*ys, 
    1600*ys, 3.8235*days, 3.10*mins, 26.8*mins, 19.9*mins,
    164.3e-6, 22.3*ys, 5.015*ys, 138.376*days]

# =============================================================================
"""So we have len(hl)=14 decaying processes"""

def fun(x, y, half_life=hl):
    dydx = np.zeros(len(half_life)+1)

    dydx[0] = -y[0]/half_life[0]
    for i in range(1,len(half_life)):
        dydx[i] = y[i-1]/half_life[i-1] - y[i]/half_life[i]
    dydx[-1] = y[-2]/half_life[-2]

    return dydx


y0    = np.zeros(len(hl)+1) 
y0[0] = 1

x0 = 0
xf = np.sum(hl) + 5*hl[0]

# t1 = time.time();
# ans_rk4 = integrate.solve_ivp( fun,[x0,xf], y0 );
# t2 = time.time();
# print('Took ',ans_rk4.nfev,' evaluations and ',t2-t1,' seconds to solve with RK4.')

print('My computer could not evaluate the system of ODEs using the Runge-Kutta method')
print()

t1 = time.time()
ans_stiff = integrate.solve_ivp( fun,[x0,xf], y0, method='Radau' )
            #integrate.solve_ivp solve system dy/dt=f(t,y), returns t and y
t2 = time.time()

print('Took ',ans_stiff.nfev,' evaluations and ',t2-t1,' seconds to solve implicitly.')
print('The implicit Runge-Kutta method works better for this case, where we have half-life times that differ by huge factors')
print()

t = ans_stiff.t
    
yPb206 = ans_stiff.y[elements.index('Pb206')]
yU238  = ans_stiff.y[elements.index('U238')]
yTh230 = ans_stiff.y[elements.index('Th230')]
yU234  = ans_stiff.y[elements.index('U234')]

# =============================================================================
"""Plots"""
trim=20

fig, axs = plt.subplots(2,2, figsize=(8,8))#, sharex=True)
fig.suptitle('Radioactive decay -- elements ratios')

axs[0,0].plot(t, yPb206/yU238, color='#8dd6a1')
axs[0,0].legend(['Pb206/U238'])

axs[1,0].plot(t, yPb206/yU238, color='#8dd6a1')
axs[1,0].legend(['Pb206/U238'])
axs[1,0].set_yscale('log')

axs[0,1].plot(t[:trim], (yTh230/yU234)[:trim], color='#c78bb0')
axs[0,1].legend(['Th230/U234'])

axs[1,1].plot(t[:trim], (yTh230/yU234)[:trim], color='#c78bb0')
axs[1,1].legend(['Th230/U234'])
axs[1,1].set_yscale('log')

for ax in axs.flat:
    ax.set(xlabel='Time (s)')
    
fig.savefig('PS2_P3.png')

    
print('The ratios do make sense. ', 
      'Looking at the log plot of Pb206/U238, we can see the "instantaneous" decay.')
print()
print('The ratio Th230/U234 also make sense. ',
      'The half life of U234 is ~0.7*10^13 s. ',
      'So the creation of Th230 slows down after this period.')