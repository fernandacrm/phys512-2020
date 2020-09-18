#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 22:53:34 2020

@author: fernandacristina
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

cosine=0
lorentzian=1

cubic_poly  =1
cubic_spline=1
rational    =0
compare     =0
rational_pinv=1
compare_pinv =1


if cosine:    
    xi=np.linspace(-np.pi/2,np.pi/2,11)
    yi=np.cos(xi)
    
    x=np.linspace(xi[1],xi[-2],101)
    y_true=np.cos(x)
    

if lorentzian:
    xi=np.linspace(-1,1,8)
    yi=1/(1+xi**2)
    
    x=np.linspace(xi[1],xi[-2],101)
    y_true=1/(1+x**2)
    

# =============================================================================    
if cubic_poly:
    
    y_poly=0*y_true
    
    for i in range(len(x)):
        ind   =np.max(np.where(np.abs(x[i]>=xi))[0])
        x_use =xi[ind-1:ind+3] 
        y_use =yi[ind-1:ind+3]
        parabs=np.polyfit(x_use,y_use,3)
        pred  =np.polyval(parabs,x[i])
        y_poly[i]=pred
        
    epoly=np.std(y_poly-y_true)
                
    plt.clf()
    plt.plot(xi,yi,'*')
    plt.plot(x,y_true)
    plt.plot(x,y_poly)
    plt.legend(['"Data"','True','Poly interp'])
    
    print('Cubic poly error is ',epoly)
    
# =============================================================================    
if cubic_spline:
    
    fspline =interpolate.interp1d(xi,yi,kind='cubic')
    y_spline=fspline(x)
    espline =np.std(y_spline-y_true)
                    
    print('Cubic spline error is ',espline)
    
    plt.clf()
    plt.plot(xi,yi,'*')
    plt.plot(x,y_true)
    plt.plot(x,y_spline)
    plt.legend(['"Data"','True','Spline interp'])

# =============================================================================    
if rational:
    def rat_fit(xr,yr,n,m):
        assert(len(xr)==n+m-1)
        assert(len(yr)==len(xr))
        mat=np.zeros([n+m-1,n+m-1])
        for i in range(n):
            mat[:,i]=xr**i
        for i in range(1,m):
            mat[:,i-1+n]=-yr*xr**i
        pars=np.dot(np.linalg.inv(mat),yr)
        p=pars[:n]
        q=pars[n:]
        return p,q
    
    def rat_eval(p,q,xr):
        top=0
        for i in range(len(p)):
            top=top+p[i]*xr**i
        bot=1
        for i in range(len(q)):
            bot=bot+q[i]*xr**(i+1)
        return top/bot  
    
    if cosine:
       n=5
       m=7
       xr=np.linspace(-np.pi/2,np.pi/2,n+m-1)
       yr=np.cos(xr)
       
    if lorentzian:
        n=4
        m=5
        xr=np.linspace(-1,1,n+m-1)
        yr=1/(1+xr**2)

    p,q=rat_fit(xr,yr,n,m)
    
    y_rat=rat_eval(p,q,x)
    
    erat=np.std(y_rat-y_true)
    print('Rational function error is ',erat)
    
    plt.clf()
    plt.plot(xr,yr,'*')
    plt.plot(x,y_true)
    plt.plot(x,y_rat)
    
    plt.legend(['"Data"','True','Rat fun interp'])
    
    if lorentzian:
        print('p_i = ',p)
        print('q_i = ',q)

# =============================================================================    
if compare:
    print('Cubic poly error is ',epoly)
    print('Cubic spline error is ',espline)
    print('Rational function error is ',erat)
    
    plt.clf()    
    plt.plot(xi,yi,'*')
    plt.plot(x,y_poly)
    plt.plot(x,y_spline)
    plt.plot(x,y_rat)
    plt.legend(['"Data"','Poly','Spline','Rat fun interp'])
    
    #plt.plot(1,epoly,'o',1,espline,'o',1,erat,'o')
    
# =============================================================================    
if rational_pinv:
    def rat_fit(xr,yr,n,m):
        assert(len(xr)==n+m-1)
        assert(len(yr)==len(xr))
        mat=np.zeros([n+m-1,n+m-1])
        for i in range(n):
            mat[:,i]=xr**i
        for i in range(1,m):
            mat[:,i-1+n]=-yr*xr**i
        pars=np.dot(np.linalg.pinv(mat),yr)
        p=pars[:n]
        q=pars[n:]
        return p,q
    
    def rat_eval(p,q,xr):
        top=0
        for i in range(len(p)):
            top=top+p[i]*xr**i
        bot=1
        for i in range(len(q)):
            bot=bot+q[i]*xr**(i+1)
        return top/bot  
    
    if cosine:
       n=5
       m=7
       xr=np.linspace(-np.pi/2,np.pi/2,n+m-1)
       yr=np.cos(xr)
       
    if lorentzian:
        n=4
        m=5
        xr=np.linspace(-1,1,n+m-1)
        yr=1/(1+xr**2)

    p,q=rat_fit(xr,yr,n,m)
    
    y_rat_pinv=rat_eval(p,q,x)
    
    erat_pinv=np.std(y_rat_pinv-y_true)
    print('Rational function (pinv) error is ',erat_pinv)

    plt.clf()    
    plt.plot(xr,yr,'*')
    plt.plot(x,y_true)
    plt.plot(x,y_rat_pinv)
    
    plt.legend(['"Data"','True','Rat fun interp'])
    
    if lorentzian:
        print('p_i = ',p)
        print('q_i = ',q)

# =============================================================================    
if compare_pinv:
    print('Cubic poly error is ',epoly)
    print('Cubic spline error is ',espline)
    # print('Rational function error is ',erat)
    print('Rational function (pinv) error is ',erat_pinv)    
    
    plt.clf()
    plt.plot(xi,yi,'*')
    plt.plot(x,y_poly)
    plt.plot(x,y_spline)
    plt.plot(x,y_rat_pinv)
    plt.legend(['"Data"','Poly','Spline','Rat fun interp'])
    