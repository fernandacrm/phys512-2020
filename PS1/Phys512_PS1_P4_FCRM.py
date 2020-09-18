"""
Created on Fri Sep 18 15:06:27 2020

@author: fernandacristina
"""


import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

# I am using R=1 (already substituted in the Efield function)
    
def Efield(u,z):
    return (z-1*u)/(1**2+z**2-2*1*z*u)**(3/2)

    
def integrate_step(fun,x1,x2,args=(),tol=1e-3):
    # print('integrating from ',x1,' to ',x2)
    x = np.linspace(x1,x2,5)
    y = fun(x, *args)
    area1 = (x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2 = (x2-x1)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    if myerr<tol:
        return area2
    else:
        xm=0.5*(x1+x2)
        a1=integrate_step(fun,x1,xm,args,tol/2)
        a2=integrate_step(fun,xm,x2,args,tol/2)
        return a1+a2


#to include z=R in the z-steps, change zs to (0,4.975,200)
zs=np.linspace(0.1,4.975,200)

E_py=np.zeros(len(zs))
error_py=np.zeros(len(zs))
E_my=np.zeros(len(zs))

for i,z in enumerate(zs):
    E_py[i]=integrate.quad(Efield,-1,1, args=(z,))[0]
    error_py[i]=integrate.quad(Efield,-1,1, args=(z,))[1]

    # print(E_py)
    
    E_my[i]=integrate_step(Efield,-1,1, args=(z,))
    # print(E_my)
 

plt.plot(zs, E_py)
plt.plot(zs, E_my)
plt.xlabel('$z\ /\ R$')
plt.ylabel('$E\  \\frac{4\pi \epsilon_0}{2\pi R^2 q} $')
plt.legend(['integrate.quad', 'my integrator'])

print('Error from integrate.quad = ',np.std(error_py))
print('(integrate.quad) - (my integrator) = ',np.std(E_py - E_my))
