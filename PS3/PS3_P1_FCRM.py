#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:40:51 2020

@author: fernandacristina
"""

import numpy as np
from matplotlib import pyplot as plt

data=np.loadtxt('dish_zenith.txt')

# z = m0 + m1*x + m2*y + m3*(x**2 + y**2)
# in matrix form,
# z= A @ m, A=[1, x, y, (x**2 + y**2)], m = [m0, m1, m2, m3]
# m = v s.inv u.T z

x,y,z = data[:,0],data[:,1],data[:,2]

A = np.zeros([len(x),4])
A[:,0]=1
A[:,1]=x
A[:,2]=y
A[:,3]=x**2+y**2

u,s,v = np.linalg.svd(A,0)

m     = v.T @ (np.diag(1/s) @ (u.T @ z))
z_pred= A @ m

error = np.std(z_pred - z)

plt.clf();
plt.plot(x, z, 'x', color='#e340ab')
plt.plot(x, z_pred, '*', color='#2b1d26' )
plt.legend(['Data','Fit'])
plt.savefig('PS3_P1.png')



print('m0, m1, m2, m3 = ',m)
print('error = ',error)

# old parameters:
a = m[3]
x0= -m[1]/(2*a)
y0= -m[2]/(2*a)
z0= -(-4*a*m[0]+m[1]**2+m[2]**2)/(4*a)
oldp=[a,x0,y0,z0]
print('a, x0, y0, z0 = ',oldp)

# covariance matrix: N = (A.T @ N.inv @ A).inv
N = np.linalg.inv(A.T @ A)
print('Covariance matrix diagonal = ',np.diag(N))

# a = m3; <n3.n3> = N[3][3]; n3 = sqrt(N[3][3])
uncert_a = np.sqrt(N[3][3])
print('Uncertainty in a = ',uncert_a)

# focal length: 
f        = 1/(4*a)
uncert_f = np.abs(uncert_a/(4*a**2))
print('Focal length = ',f*1e-3,'+/-',uncert_f*1e-3)
