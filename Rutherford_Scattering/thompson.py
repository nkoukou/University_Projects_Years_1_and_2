# -*- coding: utf-8 -*-
# Thomson's model

# Packages
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.integrate as spi

# Physical constants

k=8.99e9   # N m^2 C^-2 - Coulomb's constant
q=1.60e-19 # C          - Electron's charge
u=1.66e-27 # kg         - Atomic mass unit

a=1e-12    # m          - Arbitrary length unit (not a physical constant)
r=2.88e2 #            - Separation between the centres of gold atoms
D=1.35e2 #            - Diameter of gold atom
th=8.e4  #            - Thickness of gold leaf

T=5.       # Mev        - Kinetic energy of an α-particle
Z=79       #            - Atomic number of gold
M=196.97   # u          - Mass of gold atom/nucleus
m=4.00     # u          - Mass of α-particle

xi=-500    # a          - Initial x-coordinate of α-particle
vyi=0      #            - Initial velocity along y-axis

# Parameters used to make measured quantities dimensionless
C=2*k*Z*q**2/(m*u)#     - coefficient of acceleration in differential equation 
a=1e-12           # m   - Arbitrary length unit
tau=(a**3/C)**0.5 # s   - Arbitrary time unit
v_a=a/tau         # m/s - Arbitrary speed unit

# Plots of gold atom
atom=plt.Circle((0,0),radius=D/2,fc='y',alpha=1)

'''
x=(r**2-(r/2)**2)**0.5 # pm
rho=(m*1.66e-27)/(0.5*x*r*th*1.e-36) # kg m^-3


for n in range(5):
    for y in range(20):
        if n%2==0:
            Au=plt.Circle((n*x,r*y),radius=D/2,fc='y',alpha=1)
            pl.figure(1, figsize=(10,10))
            pl.axis([-50*D, 50*D, -50*D, 50*D])
            plt.gca().add_patch(Au)
        else:
            Au=plt.Circle((n*x,r*y+r/2),radius=D/2,fc='y',alpha=1)
            pl.figure(1, figsize=(10,10))
            pl.axis([-50*D, 50*D, -50*D, 50*D])
            plt.gca().add_patch(Au)
'''

# this function sets up the coupled differential equations
# R,t are the arguments of the function
# - R is a one dimensional array with four elements
# - t is time
def F(R,t):
    x=R[0]              # distance from centre of atom in the x-axis
    y=R[1]              # distance from centre of atom in the y-axis
    d=(x**2+y**2)**0.5  # distance from centre of atom
    vx=R[2]             # velocity in the x-axis
    vy=R[3]             # velocity in the y-axis
    if d<=D/2:
        ax=8*x/D**3       # acceleration in the x-axis
        ay=8*y/D**3       # acceleration in the y-axis
    else:
        ax=x/d**3       # acceleration in the x-axis
        ay=y/d**3       # acceleration in the y-axis
    return [vx,vy,ax,ay]

# Time interval of α-particle's orbit
t=sp.linspace(0.,400,1e4)


theta=[] # Scattering angle
b=[]     # Impact parameter
dmin=[]  # Minimum distance from the nucleus
    
for i in range(100):
    
    vi=(2*T*q*1.e6/(m*u))**0.5/v_a # Total initial velocity of α-particle
    Ri=[xi,i,vi,vyi] # Initial conditions for the differential equations
    Rf=spi.odeint(F,Ri,t) # Solves the coupled differential equations
    
    # Recalling the final physical quantities
    x=Rf[:,0]
    y=Rf[:,1]
    d=(x**2+y**2)**0.5
    vx=Rf[:,2]
    vy=Rf[:,3]
    v=(vx**2+vy**2)**0.5
    
    angle=np.arccos(vx[-1]/v[-1])
    theta.append(angle)
    b.append(Ri[1])
    dmin.append(np.amin(d))
    
    # plots trajectory
    pl.figure(1, figsize=(7,7))
    plt.gca().add_patch(atom)
    pl.plot(x,y)
    pl.xlabel("x (a)", size=20)
    pl.ylabel("y (a)", size=20)

pl.figure(2, figsize=(7,7))
pl.plot(b,theta)
pl.xlabel("Impact parameter (a)", size=20)
pl.ylabel("Scattering angle (rad)", size=20)
