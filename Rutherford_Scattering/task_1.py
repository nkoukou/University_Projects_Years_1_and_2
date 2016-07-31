# Task 1

import scipy as sp
import numpy as np
import pylab as pl
import scipy.integrate as spi


# constants
m=2 # mass is 2kg
k=5 # spring constant is 5N/m

def f(X,t):
    x=X[0] # x is the displacement from equilibrium in [m]
    v=X[1] # v is the velocity of the mass in [m/s]
    a=-(k/m)*X[0] # a is the acceleration of the mass in [m/s/s]
    return [v,a]
# returns differentiated values for x and v

t=sp.linspace(0.,10.,100) # t is time in [s]
# solving every tenth of a second

X0=[0,10] # initial values for x and v

Xf=spi.odeint(f,X0,t) # solves the SHM differential equation

x=Xf[:,0]
v=Xf[:,1]
a=-(k/m)*x
#seperates the data for x, v and a

pl.figure(1)
pl.plot(t,x)
pl.xlabel("Time (s)")
pl.ylabel("Displacement (m)")
#plots x against t

pl.figure(2)
pl.plot(t,v)
pl.xlabel("Time (s)")
pl.ylabel("Velocity (m/s)")
#plots v against t

pl.figure(3)
pl.plot(t,a)
pl.xlabel("Time (s)")
pl.ylabel("Acceleration (m/s/s)")
#plots a against t

pl.figure(4)
pl.plot(x,a)
pl.xlabel("Displacement (m)")
pl.ylabel("Acceleration (m/s/s)")
#plots a against x
