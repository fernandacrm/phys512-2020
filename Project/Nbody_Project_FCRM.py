#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:46:25 2020

@author: fernandacristina
"""

"""
3D n-body
Calculate force by computing potential
1) convolve density with softened potential from a single particle 
   to calculate potential
2) take gradient of potential to calculate the force
*) use leapfrog solver with fixed timestep
"""

import numpy as np
from matplotlib import pyplot as plt
import time


save_figs = 0
# =============================================================================
plt.rcParams['figure.figsize'      ] = 5, 3
plt.rcParams['figure.titlesize'    ] = 'medium'
plt.rcParams['legend.fontsize'     ] = 12
plt.rcParams['legend.borderaxespad'] = 0.7
plt.rcParams['legend.frameon'      ] = True
plt.rcParams['legend.framealpha'   ] = 0.7
plt.rcParams['legend.markerscale'  ] = 1
plt.rcParams['legend.labelspacing' ] = 0.2
plt.rcParams['legend.handletextpad'] = 0.3
plt.rcParams['legend.handlelength' ] = 1.0
plt.rcParams['lines.linewidth'     ] = 1.5
plt.rcParams['lines.markersize'    ] = 10
plt.rcParams['axes.labelsize'      ] = 11  # fontsize of the x and y labels
plt.rcParams['xtick.labelsize'     ] = 11  # fontsize of the tick labels
plt.rcParams['ytick.labelsize'     ] = 11  # fontsize of the tick labels
plt.rcParams['xtick.direction'     ] = 'in'
plt.rcParams['ytick.direction'     ] = 'in'
plt.rcParams['axes.prop_cycle'     ] = plt.cycler('color',[ '#AD5D5D', '#F4B9A9', '#DEBAB9','#6E7E75', '#A9A570'])
plt.rcParams['figure.constrained_layout.use'] = True
# =============================================================================

# =============================================================================
def greens(m,soft):
    #get the potential from a point charge at (0,0,0)
    dx = np.arange(m)/m                   
    dx[m//2:] = dx[m//2:]-m/m             
    xmat,ymat,zmat = np.meshgrid(dx,dx,dx)  
    
    r = np.sqrt(xmat**2+ymat**2+zmat**2)
    r[r<soft] = soft

    pot = np.zeros([m,m,m])             
    pot = G/r
    return pot


def density(x,y,z,m):
    points = (x,y,z)
    grid_min = 0
    grid_max = m
    m_grid   = m
    H, edges = np.histogramdd(points, bins=m_grid, 
                              range=((grid_min, grid_max), 
                                     (grid_min, grid_max), 
                                     (grid_min, grid_max)) )
    edges_x = edges[0]
    edges_y = edges[1]
    edges_z = edges[2]

    return H, edges_x, edges_y, edges_z

# =============================================================================
# 1) convolve density with greens to get potential:
def get_pot(x,y,z,m):
    density_fft = np.fft.fftn( density(x,y,z,m)[0] )
    greens_fft  = np.fft.fftn( np.fft.fftshift(greens(m,soft)) )

    potential   = np.real( np.fft.ifftn( greens_fft * density_fft ))
    potential   = np.roll(potential,1,(0,1,2))
    
    # potential   = np.fft.fftshift(potential)
    return potential

def grad(pot):
    gy = (np.roll(pot, -1, axis=0) - np.roll(pot, 1, axis=0)) / (2 * dx) 
    gx = (np.roll(pot, -1, axis=1) - np.roll(pot, 1, axis=1)) / (2 * dx)
    gz = (np.roll(pot, -1, axis=2) - np.roll(pot, 1, axis=2)) / (2 * dx)
    return np.array([gx,gy,gz])

def get_forces(x,y,z,m):
    forces = -grad(get_pot(x,y,z,m))
    # forces  = np.gradient( get_pot(x,y,z,m) )

    return forces



# =============================================================================
fxs=[];fys=[];fzs=[]
fxs_particles=[];fys_particles=[];fzs_particles=[]

def take_step(x,y,z,vx,vy,vz,dt,m):
    xx = x+0.5*vx*dt
    yy = y+0.5*vy*dt
    zz = z+0.5*vz*dt

    periodic = 1
    if periodic:
        xx[xx<=0] = xx[xx<=0]%m
        xx[xx>=m] = xx[xx>=m]%m
        yy[yy<=0] = yy[yy<=0]%m
        yy[yy>=m] = yy[yy>=m]%m
        zz[zz<=0] = zz[zz<=0]%m
        zz[zz>=m] = zz[zz>=m]%m

    fy,fx,fz = get_forces(xx,yy,zz,m)

    fx[abs(fx)<1e-10]=0
    fy[abs(fy)<1e-10]=0
    fz[abs(fz)<1e-10]=0
    
    fxs.append(fx);         fys.append(fy);         fzs.append(fz)

    bins = np.arange(0,m)
    particle_x_bins = np.digitize(x, bins, right=True)
    particle_y_bins = np.digitize(y, bins, right=True)
    particle_z_bins = np.digitize(z, bins, right=True)

    fx_particles = np.zeros(len(x))
    fy_particles = np.zeros(len(y))
    fz_particles = np.zeros(len(z))
    
    for i in range(len(x)):
        fx_particles[i] = fx[particle_x_bins[i], particle_y_bins[i], particle_z_bins[i]]
        fy_particles[i] = fy[particle_x_bins[i], particle_y_bins[i], particle_z_bins[i]]
        fz_particles[i] = fz[particle_x_bins[i], particle_y_bins[i], particle_z_bins[i]]

        print('fx=_particles'+str(fx_particles));   print('fy_particles='+str(fy_particles));   print('fz_particles='+str(fz_particles))
        fxs_particles.append(fx_particles);         fys_particles.append(fy_particles);         fzs_particles.append(fz_particles)

    vvx = vx+0.5*dt*fx_particles
    vvy = vy+0.5*dt*fy_particles
    vvz = vz+0.5*dt*fz_particles
        
    x = x+dt*vvx
    y = y+dt*vvy
    z = z+dt*vvz

    vx = vx+dt*fx_particles
    vy = vy+dt*fy_particles
    vz = vz+dt*fz_particles
    
    return x,y,z,vx,vy,vz
# =============================================================================
m = 50
G = 50
soft=.05
# dt=soft**1.5*.2
dx = .04
dt=.01

# =============================================================================
part1 = 0

if part1:       # one particle at rest in 3D
    x, y, z  = np.ones([3,1])*25
    vx,vy,vz = np.zeros([3,1])
    
    xs=[]; ys=[]; zs=[]
    pos_evol=[]
    
    plot_1body=0
    if plot_1body:
        for i in range(100):
            x,y,z,vx,vy,vz = take_step(x,y,z,vx,vy,vz,dt,m)
            xs.append(np.real(x))
            ys.append(np.real(y))
            zs.append(np.real(z))
            pos_evol.append([np.real(x)[0],np.real(y)[0],np.real(z)[0] ])
    
            plt.plot(x,y,'r*')
            plt.axis([15,35,15,35])
            plt.pause(.001)
            plt.title('1 body at rest, dt='+str(dt)+', soft='+str(soft)+
                      ', steps='+str(i))
            plt.xlabel('x');plt.ylabel('y')

    print(pos_evol)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xs,ys,zs)

    minor_ticks_x = density(x,y,z,m)[1]
    minor_ticks_y = density(x,y,z,m)[2]
    minor_ticks_z = density(x,y,z,m)[3]
 
    ax.set(xlim=(15,35), 
            ylim=(15,35),
            zlim=(15,35) )
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    plt.title('One particle at rest')
    
    if save_figs:
        plt.savefig('Project_1_3D.png', dpi=100)

# =============================================================================
part2 = 1

if part2:       # two particles in circular orbit in 3D
    x,y,z = [[20,30],[25,25],[25,25]]

    check_forces=0
    if check_forces:
      
        x,y,z = [[25],[25],[25]]

        forces= get_forces(x,y,z,m)

        xspace = np.linspace(0,49,50)
        yspace = np.linspace(0,49,50)
        zspace = np.linspace(0,49,50)
 
        plt.figure(figsize=(10,10))
        plt.quiver(xspace,yspace,forces[0,:,:,25],forces[1,:,:,25])
        plt.xlabel('x');plt.ylabel('y')    
    
        plt.figure(figsize=(10,10))
        plt.quiver(xspace,zspace,forces[0,:,25,:],forces[2,:,25,:])     
        plt.xlabel('x');plt.ylabel('z')    
    
        plt.figure(figsize=(10,10))
        plt.quiver(xspace,zspace,forces[1,:,25,:],forces[2,:,25,:])     
        plt.xlabel('y');plt.ylabel('z')    
    
    vx = np.array([0,0])
    vy = np.array([-17.5,17.5])
    vz = np.array([0,0])

    xs=[]; ys=[]; zs=[]
    pos_evol=[]


    plot_2body=1
    if plot_2body:
        for i in range(200):
            x,y,z,vx,vy,vz = take_step(x,y,z,vx,vy,vz,dt,m)
            xs.append(np.real(x))
            ys.append(np.real(y))
            zs.append(np.real(z))
            pos_evol.append([np.real(x)[0],np.real(y)[0],np.real(z)[0] ])
    
            plt.plot(x,y,'r*')
            plt.axis([15,35,15,35])
            plt.pause(.001)
            plt.title('2 bodies, dt='+str(dt)+', soft='+str(soft)+
                      ', steps='+str(i))
            plt.xlabel('x');plt.ylabel('y')
            
    print(pos_evol)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xs,ys,zs)

    minor_ticks_x = density(x,y,z,m)[1]
    minor_ticks_y = density(x,y,z,m)[2]
    minor_ticks_z = density(x,y,z,m)[3]
 
    ax.set(xlim=(15,35), 
            ylim=(15,35),
            zlim=(15,35) )
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    plt.title('Two particles in orbit')
    
    if save_figs:
        plt.savefig('Project_2.png', dpi=100)
    
# =============================================================================
# =============================================================================
part3 = 0

if part3:       # two particles in circular orbit in 3D
    n = 10**5
    position = m*np.random.rand(n,3)
    x = position[:,0]
    y = position[:,1]
    z = position[:,2]


    v = m*np.random.rand(n,3)
    vx = position[:,0]
    vy = position[:,1]
    vz = position[:,2]

    xs=[]; ys=[]; zs=[]
    pos_evol=[]


    plot_2body=1
    if plot_2body:
        for i in range(200):
            x,y,z,vx,vy,vz = take_step(x,y,z,vx,vy,vz,dt,m)
            xs.append(np.real(x))
            ys.append(np.real(y))
            zs.append(np.real(z))
            pos_evol.append([np.real(x)[0],np.real(y)[0],np.real(z)[0] ])
    
            plt.plot(x,y,'r*')
            plt.axis([15,35,15,35])
            plt.pause(.001)
            plt.title('2 bodies, dt='+str(dt)+', soft='+str(soft)+
                      ', steps='+str(i))
            plt.xlabel('x');plt.ylabel('y')
            
    print(pos_evol)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xs,ys,zs)

    minor_ticks_x = density(x,y,z,m)[1]
    minor_ticks_y = density(x,y,z,m)[2]
    minor_ticks_z = density(x,y,z,m)[3]
 
    ax.set(xlim=(15,35), 
            ylim=(15,35),
            zlim=(15,35) )
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    plt.title('Two particles in orbit')
    
    if save_figs:
        plt.savefig('Project_2.png', dpi=100)
    


# =============================================================================
check_grid = 1
if check_grid:
    H=density(x,y,z,m)[0]
    edges_x=density(x,y,z,m)[1]
    edges_y=density(x,y,z,m)[2]
    fig,ax = plt.subplots(figsize=(5,5), dpi=100)
    ax.imshow(
        H.sum(axis=2),
        origin="lower",
        extent=(edges_y.min(), edges_y.max(), edges_x.min(), edges_x.max()), 
        aspect="auto" )
    
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    minor_ticks_x = edges_x
    minor_ticks_y = edges_y
    ax.set(xlim=(minor_ticks_y[0], minor_ticks_y[-1]), ylim=(minor_ticks_x[0], minor_ticks_x[-1]))
    ax.set_xticks(minor_ticks_y, minor=True)
    ax.set_yticks(minor_ticks_x, minor=True)
    ax.grid(which='both', alpha=0.75, color='w')
    plt.show()
    
    check=H.sum(axis=2)
    # print(check)

