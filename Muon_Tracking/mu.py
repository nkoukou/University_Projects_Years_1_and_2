# Initial Version Created by S. Zenz, Imperial College
# Developed by S. Mousafeiris and N. Koukoulekidis, Imperial College
# May 2015
# A starting script to draw the CMS solenoid in 2d and 3d and plot a track and hits

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Features of CMS detector
B = 3.8 # Magnetic field strength in 1st solenoid, units: Tesla
B2 = 2.0 # Magnetic field strength in 2nd solenoid, units: Tesla. #~ Approximate, to be improved
r_cal = 1.950 # Inner radius of calorimeter, units: meters
r_sol = 2.950 # Radius of 1st solenoid, units: meters
r_sol2 = 7.500 # Radius of 2nd solenoid, units: meters
z_sol = 1000000.00 # Length of solenoid, units: meters
E_loss = -1.5 # Rate of energy loss with respect to distance in calorimeter, units : GeV/meters
E_loss_err = 0.5 # Standard deviation of energy loss, units : GeV/meters
muon_mass = 0.106 # Mass of muon, units: GeV/c^2

# Tracker layers (z = 0) [units: m]
#~ Approximate locations, to be improved.
r_pixel = [0.04,0.07,0.11]
r_tib=[0.26,0.32,0.43,0.62]
r_tob=[0.71,0.79,0.88,0.97,1.07,1.14]
r_tracker=np.append(np.append(r_pixel,r_tib), r_tob)


r_mb1= [4.20,4.22,4.24,4.26,4.42,4.44,4.46,4.48]
r_mb2= [5.00,5.02,5.04,5.06,5.24,5.26,5.28,5.30]
r_mb3= [6.08,6.10,6.12,6.14,6.32,6.34,6.36,6.38]
r_mb4= [7.10,7.12,7.14,7.16,7.34,7.36,7.38,7.40]

r_drift=np.append(np.append(r_mb1,r_mb2),np.append(r_mb3,r_mb4))


r_layers = np.append(r_tracker, r_drift)

# Initial particle momentum
# pz = momentum in z direction, units: GeV/c
# pt  = transverse momentum, units: GeV/c

# initial particle position
# x0 = initial x position, units: meters
# y0 = initial y position, units: meters
# z0 = initial z position, units: meters

# phi0 = initial angle of particle in transverse plane, units : rads

# Sense of direction determined by charge 
# q = -1 muon (counterclockwise) , q = 1 antimuon (clockwise)



# Plots particle path and tracker hits in crosssection of detector.
def plot_tracker_path(x_path, y_path, z_path, xhit, yhit, zhit):
    # Create 2d axes
    fig2 = plt.figure(2, figsize=((np.pi)**2,(np.pi)**2))
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_xlim(-1.1*r_sol2,1.1*r_sol2)
    ax2.set_ylim(-1.1*r_sol2,1.1*r_sol2)
    
    solenoid_points(ax2)
    plot_path(x_path, y_path, z_path, xhit, yhit, zhit, ax2)
	

# Plots particle path and tracker hits in detector, 3D plot.	
def plot_tracker_path_3D(x_path, y_path, z_path, xhit, yhit, zhit):
    # Create 3D axes
    fig3 = plt.figure(1)
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.set_zlabel("Z [m]")
    ax3.set_xlim3d(-1.1*r_sol2,1.1*r_sol2)
    ax3.set_ylim3d(-1.1*r_sol2,1.1*r_sol2)
    ax3.set_zlim3d(-1.1*z_sol,1.1*z_sol)
    
    solenoid_points_3D(ax3)
    plot_path_3D(x_path, y_path, z_path, xhit, yhit, zhit, ax3)


# Creates the detector in 2D.
def solenoid_points(ax2):
    # Create solenoid points
    x_grid=np.linspace(-r_sol, r_sol, 100)
    y_grid = np.sqrt(r_sol*r_sol-x_grid*x_grid)
    
    # Create 2nd solenoid points
    x_grid2=np.linspace(-r_sol2, r_sol2, 200)
    y_grid2 = np.sqrt(r_sol2*r_sol2-x_grid2*x_grid2)

    # Create Calorimeter points
    x_grid3=np.linspace(-r_cal, r_cal, 200)
    y_grid3 = np.sqrt(r_cal*r_cal-x_grid3*x_grid3)
    
    # Plot solenoid points in 2d
    ax2.plot(x_grid,y_grid,color='b',label='Solenoid Magnet')
    ax2.plot(x_grid,-y_grid,color='b')
    
    # Plot 2nd solenoid points in 2d
    ax2.plot(x_grid2,y_grid2,color='c',label='CMS Outer Edge')
    ax2.plot(x_grid2,-y_grid2,color='c')
    
    # Plot CMS points in 2d
    ax2.plot(x_grid3,y_grid3,color='g',label='Calorimeter Inner Edge')
    ax2.plot(x_grid3,-y_grid3,color='g')


# Creates the detector in 3D.		
def solenoid_points_3D(ax3):
    # Create solenoid points
    x_grid=np.linspace(-r_sol, r_sol, 100)
    z_grid=np.linspace(-z_sol, z_sol, 100)
    Xc, Zc=np.meshgrid(x_grid, z_grid)
    Yc = np.sqrt(r_sol*r_sol-Xc*Xc)
    y_grid = np.sqrt(r_sol*r_sol-x_grid*x_grid)
    
    # Create 2nd solenoid points
    x_grid2=np.linspace(-r_sol2, r_sol2, 200)
    z_grid2=np.linspace(-z_sol, z_sol, 200)
    Xc2, Zc2=np.meshgrid(x_grid2, z_grid2)
    Yc2 = np.sqrt(r_sol2*r_sol2-Xc2*Xc2)
    y_grid2 = np.sqrt(r_sol2*r_sol2-x_grid2*x_grid2)
    
    # Plot solenoid points as mesh in 3d
    ax3.plot_surface(Xc, Yc, Zc,  rstride=4, cstride=4, color='b', alpha=0.2)
    ax3.plot_surface(Xc, -Yc, Zc,  rstride=4, cstride=4, color='b', alpha=0.2)
   	
    # Plot 2nd solenoid points as mesh in 3d
    ax3.plot_surface(Xc2, Yc2, Zc2,  rstride=4, cstride=4, color='c', alpha=0.2)
    ax3.plot_surface(Xc2, -Yc2, Zc2,  rstride=4, cstride=4, color='c', alpha=0.2)


# Calculates the particle's angle with the x-axis at the point where it crosses the first part of the detector.
def phi_calc(Rc, x_fin, y_fin, phi0, x0, y0, q):
    # The ratio of tangent_up over tangent_down is the slope of the muon trajectory at the crosspoint.
    tangent_up = (x0-x_fin + q*Rc*np.cos(phi0))
    tangent_down = (y_fin - y0 - q*Rc*np.sin(phi0))
    # Systematic correction of phi1.
    if q > 0 :
        phi1 = np.arctan2(tangent_up,tangent_down) + 3*np.pi/2
    else:
        phi1 = np.arctan2(tangent_up,tangent_down) + np.pi/2
    return phi1

# Calculates the total loss in momentum of the particle due to the energy loss in the calorimeter.    
def energy_loss(Rc, x_cal, y_cal, z_cal, x_fin, y_fin, z_fin, phi0, x0, y0, q, pt, pz):
    phi_cal = phi_calc(Rc, x_cal, y_cal, phi0, x0, y0, q)
    phi1 = phi_calc(Rc, x_fin, y_fin, phi0, x0, y0, q)
    T = (phi1-phi_cal)/(2*np.pi)
    H = z_fin-z_cal
    L = np.sqrt((np.pi*(2*Rc)*T)**2 + H**2)
    p_tot = np.sqrt(pt*pt + pz*pz)
    pz_2 = (pz/p_tot)*(np.random.normal(E_loss,E_loss_err))*L + pz
    pt_2 = (pt/p_tot)*(np.random.normal(E_loss,E_loss_err))*L + pt
    return pt_2, pz_2, phi1


# Calculates the path of the particle through a layer of the detector and the tracker hits.
def path_creation(Rc, ldip, x0, y0, z0, phi0, q, pt, pz, layer, x_path, y_path, z_path, plot, iterations, crosspoint):
    # Create parameters of helix
    s = np.linspace(0, 3*np.pi*Rc, iterations)  # path length
    z = s*np.sin(ldip)  + z0
    x = x0 -q*Rc*(np.cos(phi0 -q*s*np.cos(ldip)/Rc) - np.cos(phi0))
    y = y0 -q*Rc*(np.sin(phi0 -q*s*np.cos(ldip)/Rc) - np.sin(phi0))
    # Don't plot helix beyond tracker volume
    r = np.sqrt(x*x+y*y)
    for i in range(len(r)):
        if layer == 0 and r[i]<r_cal and r[i+1]>r_cal:
            x_cal,y_cal,z_cal =x[i],y[i],z[i]
        if layer == 0 and r[i] > r_sol:
            if plot: print "Crossing tracker at %ith point which has (r,z)=(%f,%f) (x,y)=(%f,%f) z/r=%f pz/pt=%f Rc=%f" % (i,r[i],z[i],x[i],y[i],z[i]/r[i],pz/pt,Rc)
            crosspoint = i
            x_fin = x[i]
            y_fin = y[i]
            z_fin = z[i]
            x_path= np.append(x_path, x[:i])
            y_path= np.append(y_path, y[:i])
            z_path= np.append(z_path, z[:i])
            return x_cal, y_cal, z_cal, x_fin, y_fin, z_fin, x_path, y_path, z_path, crosspoint			
            break
        if layer == 0 and abs(z[i]) > z_sol:
            if plot: print "Truncating at %ith point which has (r,z)=(%f,%f) (x,y)=(%f,%f) z/r=%f pz/pt=%f Rc=%f" % (i,r[i],z[i],x[i],y[i],z[i]/r[i],pz/pt,Rc)                    
            crosspoint = i
            x_cal,y_cal,z_cal =0,0,0
            x_fin = x[i]
            y_fin = y[i]
            z_fin = z[i]
            x_path= np.append(x_path, x[:i])
            y_path= np.append(y_path, y[:i])
            z_path= np.append(z_path, z[:i])
            return x_cal, y_cal, z_cal, x_fin, y_fin, z_fin, x_path, y_path, z_path, crosspoint
            break		
        if layer == 1 and  r[i] > r_sol2 or abs(z[i]) > z_sol:
            if plot: print "Truncating at %ith point which has (r,z)=(%f,%f) (x,y)=(%f,%f) z/r=%f pz/pt=%f Rc=%f" % (i+crosspoint,r[i],z[i],x[i],y[i],z[i]/r[i],pz/pt,Rc)
            x_path= np.append(x_path, x[:i])
            y_path= np.append(y_path, y[:i])
            z_path= np.append(z_path, z[:i])
            return x_path, y_path, z_path
            break
			
	
# Creates full, continuous path of particle throughout detector.
def particle_path(x0, y0, z0, phi0, q, pt, pz, x_path, y_path, z_path, plot, iterations):
    # Calculate radius of curvature
    Rc = pt / (0.3*B) # units: meters
    #print 'Rc', Rc
    #Rc2 = pt / (0.3*B2) # units: meters
    #Calculate dip angle
    ldip = np.arctan2(pz,pt)
    layer = 0
    x_cal, y_cal, z_cal, x_fin, y_fin, z_fin, x_path, y_path, z_path, crosspoint = path_creation(Rc, ldip, x0, y0, z0, phi0, q, pt, pz, layer, x_path, y_path, z_path, plot, iterations, crosspoint=0)
    if np.sqrt(x_fin*x_fin + y_fin*y_fin) < r_sol:
        return x_path, y_path, z_path
    else :
        layer = 1
        pt2,pz2, phi1 = energy_loss(Rc, x_cal, y_cal, z_cal, x_fin, y_fin, z_fin, phi0, x0, y0, q, pt, pz)
        Rc2 = pt2/(0.3*B2) # units: meters
        #print 'Rc2', Rc2
        x_path, y_path, z_path = path_creation(Rc2, ldip, x_fin, y_fin, z_fin, phi1, -q, pt2, pz2, layer, x_path, y_path, z_path, plot, iterations, crosspoint)
        return x_path, y_path, z_path
   
   
# Calculates a list with the detector hit locations. Inefficient - to be improved.
def hit_calc(xhit, yhit, zhit, x_path, y_path, z_path, r_layers):
    r = np.sqrt(x_path*x_path+y_path*y_path)
    for i in range(len(r)):
        for rl in r_layers:
            if i>0 and r[i] > rl and r[i-1] < rl:
                xhit.append(0.5*(x_path[i]+x_path[i-1]))
                yhit.append(0.5*(y_path[i]+y_path[i-1]))
                zhit.append(0.5*(z_path[i]+z_path[i-1]))
                break
    return xhit, yhit, zhit
		

# Plots particle path and hits in cross-section of detector.
def plot_path(x_path, y_path, z_path, xhit, yhit, zhit, ax2):
    # Plot the helix and hits
    ax2.plot(x_path, y_path,label='Particle path',color='r')
    ax2.plot(np.array(xhit),np.array(yhit),label="Tracker hit (unsmeared)",color='r',marker='o',linestyle="none")
    ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    ncol=2, mode="expand", borderaxespad=0.,numpoints=1)
    ax2.set_aspect('equal')
    
    # Show the figures we've made
    plt.show()

	
# Plots particle path and hits in 3D detector.
def plot_path_3D(x_path, y_path, z_path, xhit, yhit, zhit, ax3):
    # Plot the helix and hits
    ax3.plot(x_path, y_path, z_path, label='Particle path', color='r')
    ax3.plot(xhit,yhit,zhit, label="Tracker hit (unsmeared)",color='r',marker='o',linestyle="none")
    ax3.set_aspect('equal')
    # Show the figures we've made
    plt.show()


#Optimizes line-resolution used to calculate hit-points depending on momentum.
def iter_calc(pt):
    if pt <= 100 :
        iterations = 2*10**5
    if pt > 100 and pt <= 500:
        iterations = 1*10**6        
    if pt > 500 and pt <= 1500:
        iterations = 2*10**6
    if pt > 1500 and pt <= 2500:
        iterations = 2*10**6
    if pt > 2500 and pt <= 3500:
        iterations = 3*10**6
    if pt > 3500 and pt <= 4500:
        iterations = 4*10**6
    if pt > 4500 and pt <= 5500:
        iterations = 5*10**6
    if pt > 5500 and pt <= 6500:
        iterations = 6*10**6
    if pt > 6500:
        iterations = 8*10**6
    return iterations

# Combines all previous functions, takes initial conditions of particle as arguments and plots its path in desired way.
# If plot is set to False, function only returns lists of the coordinates of hit locations.
def muon_path(x0=0., y0=0., z0=0., phi0=0., pt=4., pz=1., q=-1, do3D=False, plot=True):
    phi0 = phi0 -np.pi/2 # 90 degree conventional correction
    x_path,y_path,z_path,xhit,yhit,zhit = [], [], [], [], [], []
    iterations = iter_calc(pt)
    x_path,y_path,z_path = particle_path(x0, y0, z0, phi0, q, pt, pz, x_path, y_path, z_path, plot, iterations)
    xhit,yhit,zhit = hit_calc(xhit, yhit, zhit, x_path, y_path, z_path, r_layers)
    if plot:
        plot_tracker_path(x_path, y_path, z_path, xhit, yhit, zhit)
        if do3D: plot_tracker_path_3D(x_path, y_path, z_path, xhit, yhit, zhit)
    else: 
        return xhit, yhit, zhit
        
        
        
#=============================================#
#PARAVIEW FUNCTIONS#
def muon_path_pv(x0=0., y0=0., z0=0., phi0=0., pt=4., pz=0.5, q=-1, do3D=False, plot=True):
    phi0 = phi0 -np.pi/2 # 90 degree conventional correction
    x_path,y_path,z_path,xhit,yhit,zhit = [], [], [], [], [], []
    iterations = iter_calc(pt)
    x_path,y_path,z_path = particle_path(x0, y0, z0, phi0, q, pt, pz, x_path, y_path, z_path, plot, iterations)
    xhit,yhit,zhit = hit_calc(xhit, yhit, zhit, x_path, y_path, z_path, r_layers)
    plot_tracker_path_3D(x_path, y_path, z_path, xhit, yhit, zhit)
    return x_path,y_path,z_path
    
import csv

def create_file(pt,pz,phi0):
    csvfile = "DATA/datax.csv"
    csvfile2 = "DATA/datay.csv"
    csvfile3 = "DATA/dataz.csv"
    x_data,y_data,z_data = muon_path_pv(pt=pt, pz=pz, phi0=phi0)
    data = zip(x_data,y_data,z_data)
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in x_data:
            writer.writerow([val])
    with open(csvfile2, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in y_data:
            writer.writerow([val])
    with open(csvfile3, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in z_data:
            writer.writerow([val])
            
            
#=============================================#
#IMPORTING DATA#

def import_data():
    raw = np.fromfile('C:\Users\Nikos\Documents\GitHub\Muon-Tracking\muon_list.txt', sep=' ')
    raw_split = np.split(raw,40000)
    return raw_split



    
        
