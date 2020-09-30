#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 01:39:26 2020

@author: fernandacristina
"""

import numpy as np

def my_fun(x):
    return 1/(1+x**2)

class_int= 1
my_int   = 1

# =============================================================================
if class_int:
    def integrator(fun, a, b, tol):
        # print('integrating from ',a,' to ',b)
        x = np.linspace(a,b,5)
        y = fun(x)

        area1 = (b-a)*(y[0]+4*y[2]+y[4])/6
        area2 = (b-a)*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12

        myerr = np.abs(area1-area2)

        if myerr<tol:
            return area2
        else:
            xm = 0.5*(a+b)
            a1 = integrator(fun, a, xm, tol/2)
            a2 = integrator(fun, xm, b, tol/2)
            return a1+a2
            
    ans_class=integrator(my_fun,-10,10,0.0001)
    print('The integral of my func using the old integrator is ', ans_class,'.')
 
# =============================================================================
if my_int:    
    def integrator(fun, a, b, tol, index, x, y, count):
        # print('integrating from ',a,' to ',b)
        
        if index ==0:       
            x=np.linspace(a,b,5)
            y=fun(x)

        else:
            xx=np.linspace(a,b,5)
            yy=[]
        
            for n in range(len(xx)):
                called=0
                
                for old_i in range(len(x)):
                    if xx[n]==x[old_i]:
                        yy.append(y[old_i])
                        
                        count =count+1
                        called=1
 
                if called==0:
                    yy.append(fun(xx[n]))

            x=xx
            y=yy

        area1 = (b-a)*(y[0]+4*y[2]+y[4])/6
        area2 = (b-a)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
        myerr = np.abs(area1-area2)
        index = index+1

        if myerr<tol:
            return area2, count
        else:
            xm = 0.5*(a+b)
            a1 = integrator(fun, a, xm, tol/2, index, x, y, count)[0]
            a2 = integrator(fun, xm, b, tol/2, index, x, y, count)[0]
 
            c1 = integrator(fun, a, xm, tol/2, index, x, y, count)[1]
            c2 = integrator(fun, xm, b, tol/2, index, x, y, count)[1]
            return a1+a2, c1+c2
 
    ans=integrator(my_fun,-10,10,0.0001,0,[],[],0)
    print('The integral of my func is ', ans[0],'. It saves ',ans[1],' function calls.')
 
# =============================================================================
    """Try some other functions"""
    ans_sin=integrator(np.sin,0,np.pi,0.0001,0,[],[],0)
    print('The integral of sin(x) from 0 to pi is ', 
          ans_sin[0],'. It saves ',ans_sin[1],' function calls.')
    
    ans_exp=integrator(np.exp,0,1,0.0001,0,[],[],0)
    print('The integral of exp(x) from 0 to 1 is ', 
          ans_exp[0],'. It saves ',ans_exp[1],' function calls.')
