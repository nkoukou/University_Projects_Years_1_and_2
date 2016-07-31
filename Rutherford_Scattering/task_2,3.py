# Task 2 and Task 3

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.integrate as spi

"""
Assumptions

1. The orbits of Mars and Earth are coplanar
2. Mars is at rest
3. M_S << M_M

"""
# constants
G=6.67e-11 # gravitational constant is 6.67*10^(-11) m^3kg^(-1)s^(-2)
M_M=6.4e23
R_M=3.4e6
M_S=260

Mars=plt.Circle((0,0),radius=R_M,fc='r',alpha=1)

# sets up the coupled differential equations
def F(R,t):
    x=R[0]
    y=R[1]
    r=(x**2+y**2)**0.5
    vx=R[2]
    vy=R[3]
    ax = -G*M_M*x/r**3
    ay = -G*M_M*y/r**3
    return [vx,vy,ax,ay]

###
tf=1e6
t=sp.linspace(0.,tf,1e3) # t is time in [s]

j=2
xi=[6*R_M, 6*R_M, 3*R_M,-7*R_M]
yi=[8*R_M, 8*R_M, 4*R_M,2*R_M]
vxi=[100, 1000,-1500,1700]
vyi=[505.5, 505.5, 1660,1500]
ri=(xi[j]**2+yi[j]**2)**0.5
vi=(vxi[j]**2+vyi[j]**2)**0.5

Ri=[xi[j],yi[j],vxi[j],vyi[j]]
Rf=spi.odeint(F,Ri,t) # solves the coupled differential equations

# definitions of the quantities used
x=Rf[:,0]
y=Rf[:,1]
r=(x**2+y**2)**0.5
vx=Rf[:,2]
vy=Rf[:,3]
v=(vx**2+vy**2)**0.5
ax=-G*M_M*x/r**3
ay=-G*M_M*y/r**3

theta=np.arccos(np.dot(vi,v)/(np.linalg.norm(vi)*np.linalg.norm(v)))

a=(ax**2+ay**2)**0.5
E_k=0.5*M_S*v**2
E_p=-G*M_M*M_S/r
E=E_k+E_p
#seperates the data for x, v and a

'''
if np.amin(r)<R_M:
    print "The satellite crushes on Mars."
elif np.amin(r)>R_M:
    if np.amax(r)<r[-1]:
        print "The satellite is in orbit around Mars."
    elif np.amax(r)==r[-1]:
        print "The satellite escapes Mars."
'''

pl.figure(1, figsize=(7,7))
pl.axis([0, tf, np.amin(r), np.amax(r)])
pl.plot(t,r)
pl.xlabel("Time (s)")
pl.ylabel("r (m)")
#plots r against t

pl.figure(2, figsize=(7,7))
pl.axis([0, tf, np.min(E_k), np.max(E_k)])
pl.plot(t,E_k)
pl.xlabel("Time (s)")
pl.ylabel("E_k (J)")
#plots E_k against t

pl.figure(3, figsize=(7,7))
pl.axis([0, tf, np.amin(E_p), np.amax(E_p)])
pl.plot(t,E_p)
pl.xlabel("Time (s)")
pl.ylabel("E_p (J)")
#plots E_p against t

pl.figure(4, figsize=(7,7))
pl.axis([0, tf, np.amin(E),np.amax(E_p)])
pl.plot(t,E)
pl.xlabel("Time (s)")
pl.ylabel("E (J)")
#plots E against t

pl.figure(5, figsize=(7,7))
pl.axis([0, tf, np.amin(E_p),np.amax(E_k)])
pl.plot(t,E_k)
pl.plot(t,E_p)
pl.plot(t,E)
pl.xlabel("Time (s)")
pl.ylabel("E, E_k, E_p (J)")
#plots E, E_p, E_k against t

pl.figure(6, figsize=(7,7))
pl.axis([np.amin(x), np.amax(x), np.amin(y), np.amax(y)])
plt.gca().add_patch(Mars)
pl.plot(x,y)
pl.xlabel("x (m)")
pl.ylabel("y (m)")
#plots trajectory

pl.figure(7, figsize=(7,7))
pl.axis([0, tf, np.amin(theta), np.amax(theta)])
pl.plot(t,theta)
pl.xlabel("t (s)")
pl.ylabel("theta (rad)")
# plots angle between initial and final velocity against time

'''
pl.figure(7, figsize=(7,7))
pl.axis([np.amin(v), np.amax(v), np.amin(theta), np.amax(theta)])
pl.plot(v,theta)
pl.xlabel("v (m/s)")
pl.ylabel("theta (rad)")
# plots angle between initial and final velocity against time
'''
