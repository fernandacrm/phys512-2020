import numpy as np
from matplotlib import pyplot as plt

# =============================================================================
"""Make a Chebyshev fit"""

def cheb_mat(npt,ord):
    #cheb is defined from -1 to 1
    #npt is the number of points for the fit
    
    x = np.linspace(-1,1,npt)
    
    T = np.zeros([npt,ord+1])
    T[:,0] = 1.0
    
    if ord>0:
        T[:,1] = x
        
    if ord>1:
        for i in range(1,ord):
            T[:,i+1] = 2*x*T[:,i] - T[:,i-1]
    return T,x

npts= 101
ord = 8
T,x = cheb_mat(npts,ord)
# print(T,x)

"""
x is a vector giving the number of points to be fitted in the linear space from -1 to 1, 
size [1,npts] (here is [1,101]).
T is the T_n matrix, size [npts,ord] (here is [101,8])

I have to shift the points to the required range (.5,1) 
So the new x will be a linear space(.5,1,npts)
"""

x_shift = (x+3)/4
y = np.log2(x_shift)
""" 
y is a column vector, y = T_n*c_n
The cheby coeffs can then be found as c_n=(T_n)^-1*y
""" 
cheb_coeffs = np.linalg.pinv(T)@y

"Now finally fit y = T_n*c_n, and truncate:" 
cheb_fit      = T @ cheb_coeffs
trun_cheb_fit = T[:,:ord]@cheb_coeffs[:ord]

# =============================================================================
"""Make a Legendre fit"""

def leg_mat(ord,npt):
    assert(npt>0)
    assert(ord>=0)
    
    x = np.linspace(-1,1,npt)
    L = np.zeros([ord+1,npt])
    L[0,:] = 1.0    
    L[1,:] = x
    if (ord>1):
        for i in range(1,ord):
            L[i+1,:]= ( (2*i+1)*x*L[i,:] - i*L[i-1,:] )/(i+1.0)
    return L

L = leg_mat(ord,npts)
#print(L)

leg_coeffs = np.polynomial.legendre.legfit(x,y,ord) 
"legfit returns the coefficients of the Legendre series" 

leg_fit      = np.transpose(L) @ leg_coeffs
trun_leg_fit = np.transpose(L)[:,:ord] @ leg_coeffs[:ord]

# =============================================================================
"""Plot fits"""
plt.clf();
plt.plot(x_shift, trun_cheb_fit, color='#8dd6a1', linewidth=5.0)
plt.plot(x_shift, trun_leg_fit, '-.', color='#c78bb0', linewidth=3.0)
plt.plot(x_shift, y, '.', markersize='4', color='#f7ed94')
plt.legend(['Chebyshev fit','Legendre fit','True'])
plt.xlabel('x');plt.ylabel('Log2(x)')

plt.savefig('PS2_P2_cheb_leg_fits.png')

# =============================================================================
"""Plot residuals"""
plt.clf();
# plt.plot(x_shift, cheb_fit-y, '*')
# plt.plot(x_shift, leg_fit-y, 'x')
plt.plot(x_shift, trun_cheb_fit - y, '*', color='#8dd6a1')
plt.plot(x_shift, trun_leg_fit - y, 'x', color='#c78bb0' )
plt.plot(x_shift, y - y, color='#303632', linewidth=.7)
plt.legend(['Chebyshev fit','Legendre fit'])
plt.xlabel('x');plt.ylabel('Residuals')

plt.savefig('PS2_P2_cheb_leg_residuals.png')

cheb_rms_e = np.sqrt(np.mean((trun_cheb_fit - y)**2))
cheb_max_e = np.max(np.abs(trun_cheb_fit - y))

leg_rms_e = np.sqrt(np.mean((trun_leg_fit-y)**2))
leg_max_e = np.max(np.abs(trun_leg_fit-y))

print('Cheb fit RMS error = ',cheb_rms_e,',  max error = ',cheb_max_e)
print('Leg fit RMS error = ',leg_rms_e,', max error = ',leg_max_e)

print()
print('As expected, truncated Chebyshev fit has higher RMS error and smaller max error when compared to a Legendre polynomial of the same order')

