# Scattering model

# Packages
import math as ma
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.integrate as spi

# Physical constants

k=8.99e9   # N m^2 C^-2 - Coulomb's constant
q=1.60e-19 # C          - Electron's charge
Na=6.02e23 # mol^-1     - Avocadro's constant
u=1.66e-27 # kg         - Atomic mass unit

a=1e-12    # m          - Arbitrary length unit (not a physical constant)
r=2.88e2   # a          - Separation between the centres of gold atoms
D=1.35e2   # a          - Diameter of gold atom
Dn=0.01    # a          - Diameter of gold nucleus
L=8.e4     # a          - Thickness of gold leaf

T=5.       # Mev        - Kinetic energy of an a-particle
Z=79       #            - Atomic number of gold
M=196.97   # u          - Mass of gold atom/nucleus
m=4.00     # u          - Mass of a-particle
E_an=8.e-13# J          - Analytically calculated total energy

xi=-500    # a          - Initial x-coordinate of a-particle
vyi=0      #            - Initial velocity along y-axis

theta=[]   # rad        - Scattering angle 
b=[]       # a          - Impact parameter
dmin=[]    # a          - Minimum distance from the nucleus
fraction=[]#            - Fraction of particles scattered at angle greater or equal than the scattering angle

# Parameters used to make measured quantities dimensionless
C=2*k*Z*q*q/(m*u)#    - coefficient of acceleration in differential equation 
a=1e-12            # m  - Arbitrary length unit
tau=(a**3/C)**0.5 # s  - Arbitrary time unit
v_a=a/tau         # m/s - Arbitrary speed unit

# Plots of gold atom and nucleus
atom=plt.Circle((0,0),radius=D/2,fc='y',alpha=1)
nucleus=plt.Circle((0,0),radius=Dn/2,fc='r',alpha=1)

rho=(7*m*u)/(r*a)**3 # kg m^-3
fraction_constant=(Na*L*a*rho*ma.pi*k*k*q*q*Z*Z)/(M*1.e-3*(T*1.e6)**2)

# This function sets up the coupled differential equations for Rutherford's model
# R,t are the arguments of the function
# - R is a one dimensional array with four elements
# - t is time
def F(R,t):
    x=R[0]              # distance from nucleus in the x-axis
    y=R[1]              # distance from nucleus in the y-axis
    d=(x*x+y*y)**0.5   # distance from nucleus
    vx=R[2]             # velocity in the x-axis
    vy=R[3]             # velocity in the y-axis
    ax= x/d**3          # acceleration in the x-axis
    ay= y/d**3          # acceleration in the y-axis
    return [vx,vy,ax,ay]
    
# Time interval of α-particle's orbit
t=sp.linspace(0.,500,1e5)

# Iterates for different impact parameters
for i in range(101):
    
    vi=(2*T*q*1.e6/(m*u))**0.5/v_a # Total initial velocity of a-particle
    Ri=[xi,0.01*i,vi,vyi]           # Initial conditions for the differential equations
    Rf=spi.odeint(F,Ri,t)           # Solves the coupled differential equations
    
    # Recalling the physical quantities
    x=Rf[:,0]
    y=Rf[:,1]
    d=(x*x+y*y)**0.5
    vx=Rf[:,2]
    vy=Rf[:,3]
    v=(vx*vx+vy*vy)**0.5
    
    b.append(Ri[1])
    angle=np.arccos(vx[-1]/v[-1])
    theta.append(angle)
    dmin.append(np.amin(d))
    fraction.append(fraction_constant*(1+np.cos(angle))/(1-np.cos(angle)))
    
    # plots trajectory
    pl.figure(1,figsize=(7,7))
    plt.gca().add_patch(atom)
    plt.gca().add_patch(nucleus)
    pl.plot(x,y)
    pl.xlabel("x (a)", size=20)
    pl.ylabel("y (a)", size=20)

test_one=((1+np.cos(theta))/(1-np.cos(theta)))**0.5 # Test function 1
halftheta=[0.5*j for j in theta[1:]] # The first element is excluded because of division by zero error
test_two=b[1:]*np.cos(halftheta)/(1-np.sin(halftheta)) # Test function 2
error_dmin=(dmin[1:]-test_two)/test_two # Error in minimum distance
Ek=0.5*m*u*(v*v_a)**2 # Kinetic energy in J
Ep=(2*Z*q**2)/(d*a) # Potential energy in J
E=Ek+Ep # Total energy in J
error_E=(E-E_an)/E_an # Error in total energy

# Plots only indicated graph and closes the rest
def plot(num):
    
    plot_set=[2,3,4,5,6,7]
    plot_set.remove(num)

    pl.figure(2, figsize=(7,7))
    pl.plot(theta,b)
    pl.xlabel("Scattering angle (rad)", size=20)
    pl.ylabel("Impact parameter (a)", size=20)

    pl.figure(3, figsize=(7,7))
    pl.plot(test_one,b)
    pl.xlabel("test function 1", size=20)
    pl.ylabel("Impact parameter (a)", size=20)

    pl.figure(4, figsize=(7,7))
    pl.plot(t,E)
    pl.xlabel("time (tau)", size=20)
    pl.ylabel("Energy (J)", size=20)

    pl.figure(5, figsize=(7,7))
    pl.plot(t,error_E)
    pl.xlabel("time (tau)", size=20)
    pl.ylabel("Error in Energy", size=20)

    pl.figure(6, figsize=(7,7))
    pl.plot(b[1:],error_dmin)
    pl.xlabel("Impact parameter (a)", size=20)
    pl.ylabel("Error in d_min (a)", size=20)
    
    pl.figure(7, figsize=(7,7))
    pl.plot(theta,fraction)
    pl.xlabel("Scattering angle (rad)", size=20)
    pl.ylabel("Fractional scattering", size=20)
    
    for j in plot_set:
        pl.close(j)
    pl.show()
    
    '''
# This commented out section replaces Rutherford's for Thompson's model
#  To use Thompson's model comment out right before the function for Rutherford's model until this comment
def F(R,t):
    x=R[0]              # distance from centre of atom in the x-axis
    y=R[1]              # distance from centre of atom in the y-axis
    d=(x**2+y**2)**0.5  # distance from centre of atom
    vx=R[2]             # velocity in the x-axis
    vy=R[3]             # velocity in the y-axis
    if d<=D/2:
        ax=8*x/D**3     # acceleration in the x-axis
        ay=8*y/D**3     # acceleration in the y-axis
    else:
        ax=x/d**3       # acceleration in the x-axis
        ay=y/d**3       # acceleration in the y-axis
    return [vx,vy,ax,ay]

# Time interval of α-particle's orbit
t=sp.linspace(0.,400,1e6)


theta=[] # Scattering angle
b=[]     # Impact parameter
dmin=[]  # Minimum distance from the nucleus

# Iterates for different impact parameters
for i in range(int(D/10)):
    
    vi=(2*T*q*1.e6/(m*u))**0.5/v_a # Total initial velocity of α-particle
    Ri=[xi,10*i,vi,vyi]            # Initial conditions for the differential equations
    Rf=spi.odeint(F,Ri,t)          # Solves the coupled differential equations
    
    # Recalling the physical quantities
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
'''
