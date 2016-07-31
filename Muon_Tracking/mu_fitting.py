from mu import *
from scipy import optimize,misc
import time
import sys

r_pixel = [0.04,0.07,0.11]
r_tib=[0.26,0.32,0.43,0.62]
r_tob=[0.71,0.79,0.88,0.97,1.07,1.14]
r_tracker=np.append(np.append(r_pixel,r_tib), r_tob)


r_mb1= [4.20,4.22,4.24,4.26,4.42,4.44,4.46,4.48]
r_mb2= [5.00,5.02,5.04,5.06,5.24,5.26,5.28,5.30]
r_mb3= [6.08,6.10,6.12,6.14,6.32,6.34,6.36,6.38]
r_mb4= [7.10,7.12,7.14,7.16,7.34,7.36,7.38,7.40]

r_drift=np.append(np.append(r_mb1,r_mb2),np.append(r_mb3,r_mb4))

# Errors

err_1_t=10.*10**(-6) # Pixel detector transverse error, units: meters
err_1_z=10.*10**(-6) # Pixel detector z error, units: meters
err_1 = [err_1_t, err_1_z]

err_2_t=20.*10**(-6) # Internal Silicon Strip Tracker transverse error, units: meters
err_2_z=20.*10**(-6) # Internal Silicon Strip Tracker z error, units: meters
err_2 = [err_2_t, err_2_z]

err_3_t=30.*10**(-6) # External Silicon Strip Tracker transverse error, units: meters
err_3_z=30.*10**(-6) # External Silicon Strip Tracker z error, units: meters
err_3 = [err_3_t, err_3_z]

# Incorrect values used for testing
err_4_t=100.*10**(-6) # MB transverse error, units: meters
err_4_z=150.*10**(-6) # MB z error, units: meters
err_4 = [err_4_t, err_4_z]


#Generates data (tracker hit locations) from particle with specified initial conditions.
def gen_data(x0=0., y0=0., z0=0., phi0=0., pt=3., pz=0.5, q=-1):
    xhit, yhit, zhit = np.asarray(muon_path(x0, y0, z0, phi0, pt, pz, q, plot = False))
    rhit = np.sqrt(xhit*xhit + yhit*yhit)
    phihit = np.arctan2(yhit, xhit)
    return rhit, phihit, zhit 


#Smears data by assigning each data point a value based on a normal distribution
#created using the tracking device's standard measuring error.
def smear_func(initial, final, err, rhit, phihit, zhit):
    i = initial
    f = i + final
    r_smear = rhit[i:f]
    phi_smear = np.random.normal(phihit[i:f], err[0]/rhit[i:f])
    x_smear = r_smear * np.cos(phi_smear)
    y_smear = r_smear * np.sin(phi_smear)
    z_smear = np.random.normal(zhit[i:f], err[1])
    return f, x_smear, y_smear, z_smear
    
    
#Calculates the actual hits of the particle in case its initial position is
#beyond the innermost tracker.
def determine_start(rhit):
    r_pixel_new = []
    r_tib_new = []
    r_tob_new = []
    r_drift_new = []
    for j in range(len(r_pixel)):
        if rhit[0]<(r_pixel[j]+0.01):
            r_pixel_new = np.append(r_pixel_new, r_pixel[j])
    if len(r_pixel_new)>0:
        return r_pixel_new, r_tib, r_tob, r_drift
    else:
        for j in range(len(r_tib)):
            if rhit[0]<(r_tib[j]+0.01):
                r_tib_new = np.append(r_tib_new, r_tib[j])
        if len(r_tib_new)>0:
            return r_pixel_new, r_tib_new, r_tob, r_drift   
        else:
            for j in range(len(r_tob)):
                if rhit[0]<(r_tob[j]+0.01):
                    r_tob_new = np.append(r_tob_new, r_tob[j])
            if len(r_tob_new)>0:
                return r_pixel_new, r_tib_new, r_tob_new, r_drift
            else:
                for j in range(len(r_drift)):
                    if rhit[0]<(r_drift[j]+0.01):
                        r_drift_new = np.append(r_drift_new, r_drift[j])
                return r_pixel_new, r_tib_new, r_tob_new, r_drift_new

#Smears each data point in the appropriate way (data from multiple trackers).    
def smear_data(rhit, phihit, zhit, tube):
    x_smear, y_smear, z_smear = [],[],[]
    err_layers = [err_1, err_2, err_3, err_4]
    det_layers = [[], r_pixel, r_tib, r_tob, r_drift]
    if rhit[0]>r_pixel[0]: 
        temporary = determine_start(rhit)
        det_layers = [[], temporary[0], temporary[1], temporary[2], temporary[3]]
        r_tracker2 = np.concatenate((det_layers[0], det_layers[1], det_layers[2], det_layers[3]))
    temp = [0]
    if tube == 1:
        for j in range(len(det_layers)-2):
            temp = smear_func(temp[0], len(det_layers[j+1]), err_layers[j], rhit, phihit, zhit)
            x_smear = np.append(x_smear, temp[1])
            y_smear = np.append(y_smear, temp[2])
            z_smear = np.append(z_smear, temp[3])
    if tube == 2:
        if rhit[0] > r_pixel[0]: temp = smear_func(len(r_tracker2), len(det_layers[4]), err_layers[3], rhit, phihit, zhit)
        else: temp = smear_func(len(r_tracker), len(det_layers[4]), err_layers[3], rhit, phihit, zhit)
        x_smear = np.append(x_smear, temp[1])
        y_smear = np.append(y_smear, temp[2])
        z_smear = np.append(z_smear, temp[3])
    return x_smear, y_smear, z_smear
    


#The following functions are used to create a least squares circle fit.
#======================================================================#
def calc_R(x,y, xc, yc):
    # Calculate the distance of each 2D points from the centre (xc, yc).
    return np.sqrt((x-xc)**2 + (y-yc)**2)
 
def f(c, x, y):
    # Calculate the algebraic distance between the data points and the mean circle centred at c=(xc, yc).
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()
 


def leastsq_circle(x,y):
    # Coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    #R_err    = Ri.std()/np.sqrt(len(Ri))
    #residu   = np.sum((Ri - R)**2)
    return xc, yc, R
 
def plot_data_circle(x,y, xc, yc, R):
    f = plt.figure( facecolor='white')  #figsize=(7, 5.4), dpi=72,
    plt.axis('equal')
 
    theta_fit = np.linspace(-np.pi, np.pi, 180)
 
    x_fit = xc + R*np.cos(theta_fit)
    y_fit = yc + R*np.sin(theta_fit)
    plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
    plt.plot([xc], [yc], 'bD', mec='y', mew=1)
    plt.xlabel('x')
    plt.ylabel('y')   
    # plot data
    plt.plot(x, y, 'r.', label='data', mew=1)
 
    plt.legend(loc='best',labelspacing=0.1 )
    plt.grid()
    plt.title('Least Squares Circle')
    plt.show()
    
#===================================================================#  

#Calculates transverse momentum by fitting circular path in smeared data.
def pt_calc(rhit, phihit, zhit,tube):
    x_data, y_data, z_data = smear_data(rhit, phihit, zhit, tube)
    if len(x_data)>0: xc, yc, Rc = leastsq_circle(x_data, y_data)
    if len(x_data)==0: Rc=0
    if tube == 1:
        pt = 0.3*B*Rc
    if tube == 2:
        pt = 0.3*B2*Rc
    if tube == 1:
        k = np.polyfit(r_tracker, z_data,1)[0]
    if tube == 2:
        k = np.polyfit(r_drift, z_data,1)[0]
    p = pt*np.sqrt(k*k+1)
   #print Rc
    return pt, p
    


#Iterates calculation of transverse momentum over a given amount of times,
#and calculates a mean and a standard error for the estimated transverse
#momentum for this particle.
def pt_datapoint(pt,pz, iter_num):
    #rhit, phihit, zhit = gen_data(pt=pt, pz=pz)
    pt_data,pt_data2,p_data,p_data2 = [],[],[],[]
    for i in range(iter_num):
        rhit, phihit, zhit = gen_data(pt=pt, pz=pz)
        pt_data = np.append(pt_data, pt_calc(rhit, phihit, zhit,1)[0])
        pt_data2 = np.append(pt_data2, pt_calc(rhit, phihit, zhit,2)[0])
        p_data = np.append(p_data, pt_calc(rhit, phihit, zhit,1)[1])
        p_data2 = np.append(p_data2, pt_calc(rhit, phihit, zhit,2)[1])
    p_mean = np.mean(p_data)
    p_err = (np.std(p_data))/(np.sqrt(iter_num))
    p_mean2 = np.mean(p_data2) - E_loss #Adjusting for calorimeter energy loss.
    p_err2 = (np.std(p_data2))/(np.sqrt(iter_num))
    pt_mean = np.mean(pt_data)
    pt_err = (np.std(pt_data))/(np.sqrt(iter_num))
    pt_mean2 = np.mean(pt_data2) -(pt_mean/p_mean)*E_loss #Adjusting for calorimeter energy loss.
    pt_err2 = (np.std(pt_data2))/(np.sqrt(iter_num))
    return pt_mean, pt_err, pt_mean2, pt_err2, p_mean, p_err, p_mean2, p_err2


#Iterates previous function over a given range of transverse momenta.
def pt_data(pt_i, pt_f, pz, point_num, iter_num):
    pt, pt_actual, pt_err, pt2, pt_err2, p, p_err, p2, p_err2 = [], [], [], [], [], [], [], [], []
    stepsize = float((pt_f - pt_i))/point_num
    for i in range(point_num):
        temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8 = pt_datapoint(iter_num = iter_num, pt = pt_i + i*stepsize, pz=pz)
        pt = np.append(pt, temp1)
        pt_err = np.append(pt_err, temp2)
        pt2 = np.append(pt2, temp3)
        pt_err2 = np.append(pt_err2, temp4)
        pt_actual = np.append(pt_actual, pt_i +i*stepsize)
        p = np.append(p,temp5)
        p_err = np.append(p_err,temp6)
        p2 = np.append(p2,temp7)
        p_err2 = np.append(p_err2,temp8)
        #Progress bar.
        sys.stdout.write("\r%d%%" % i)
        sys.stdout.flush()
    return pt, pt_err, pt2, pt_err2, p, p_err, p2, p_err2, pt_actual
    
#Plots the standard error in the transverse momentum and its deviation from
#the actual value against the transverse momentum.
def plot_data(pt_i=5, pt_f=20, pz=0.5, point_num=100, iter_num=50): #Does not work if muon trapped in tracker.
    pt, pt_err, pt2, pt_err2, p, p_err, p2, p_err2, pt_actual = pt_data(pt_i, pt_f, pz, point_num, iter_num)
    p_actual = np.sqrt(pt_actual*pt_actual +pz*pz)
    if all(pt) == 0: pt_dev = [0]*len(pt)
    else: pt_dev = (pt-pt_actual)
    pt2_dev = (pt2-pt_actual)
    p_dev = (p-p_actual)
    p2_dev = (p2-p_actual)
    ##Plotting transverse momentum.
    #plt.figure(1)
    #plt.subplot(211)
    #plt.title('$p_t$ error vs $p_t$, Tracker')
    #plt.ylabel('$\Delta p_t$') 
    #plt.grid()
    #plt.plot(pt_actual, pt_err/pt, 'b.' , label="$p_t$ Error")
    #plt.plot(pt_actual, pt_dev/pt, 'r.' , label="$p_t$ Deviation")
    #plt.legend(loc='best',labelspacing=0.1,numpoints=1)
    #plt.subplot(212)
    #plt.title('$p_t$ error vs $p_t$, DT')
    #plt.xlabel('$p_t$')
    #plt.ylabel('$\Delta p_t$') 
    #plt.grid()
    #plt.plot(pt_actual, pt_err2/pt2, 'b.' , label="$p_t$ Error")
    #plt.plot(pt_actual, pt2_dev/pt2, 'r.' , label="$p_t$ Deviation")
    
    
    #Plotting fractional momentum.
    plt.figure(1)
    plt.subplot(211)
    plt.ylabel('$\Delta p$') 
    plt.grid()
    plt.title('$p$ error vs $p$, Tracker')
    plt.plot(p_actual, p_err/p, 'b.' , label="$p$ Error")
    plt.plot(p_actual, p_dev/p, 'r.' , label="$p$ Deviation")
    plt.legend(loc='best',labelspacing=0.1,numpoints=1)
    plt.subplot(212)
    plt.xlabel('$p$')
    plt.ylabel('$\Delta p$') 
    plt.grid()
    plt.title('$p$ error vs $p$, DT')
    plt.plot(p_actual, p_err2/p2, 'b.' , label="$p$ Error")
    plt.plot(p_actual, p2_dev/p2, 'r.' , label="$p$ Deviation")
    plt.show()
    
    
    ##Plotting momentum.
    #plt.figure(2)
    #plt.subplot(211)
    #plt.ylabel('$\Delta p$') 
    #plt.grid()
    #plt.title('$p$ error vs $p$, Tracker')
    #plt.plot(p_actual, p_err, 'b.' , label="$p$ Error")
    #plt.plot(p_actual, p_dev, 'r.' , label="$p$ Deviation")
    #plt.legend(loc='best',labelspacing=0.1,numpoints=1)
    #plt.subplot(212)
    #plt.xlabel('$p$')
    #plt.ylabel('$\Delta p$') 
    #plt.grid()
    #plt.title('$p$ error vs $p$, DT')
    #plt.plot(p_actual, p_err2, 'b.' , label="$p$ Error")
    #plt.plot(p_actual, p2_dev, 'r.' , label="$p$ Deviation")
    #plt.show()
    
    #Plotting actual deviations.
    plt.figure(2)
    plt.ylabel('$\Delta p$') 
    plt.grid()
    plt.title('$p$ deviation from actual value')
    plt.plot(p_actual, p_dev/p_actual, 'b.' , label="$p$ Deviation, Tracker")
    plt.plot(p_actual, p2_dev/p_actual, 'r.' , label="$p$ Deviation, DT")
    plt.legend(loc='best',labelspacing=0.1,numpoints=1)
    plt.show()
    


#Plots histograms of momentum measurements for specific actual momentum, useful to deduce distributions.
def poutsoktonos(pt,pz, iter_num):
    pt_tracker_list, pt_dt_list = [],[]
  #  rhit, phihit, zhit = gen_data(pt=pt, pz=pz)
    for i in range(iter_num):
        rhit, phihit, zhit = gen_data(pt=pt, pz=pz)
        temp = pt_calc(rhit,phihit,zhit,1)
        pt_dt = pt_calc(rhit,phihit,zhit,2)[0] - (temp[0]/temp[1])*E_loss
        
        pt_tracker_list = np.append(pt_tracker_list,pt_calc(rhit,phihit,zhit,1)[0])
        pt_dt_list = np.append(pt_dt_list, pt_dt)
                
        #Progress bar.
        sys.stdout.write("\r%d%%" % i)
        sys.stdout.flush()
    bins=np.histogram(np.hstack((pt_tracker_list,pt_dt_list)), bins=100)[1]
    plt.hist(pt_tracker_list,bins=bins, color='b', alpha=1, label='Tracker Data')
    plt.hist(pt_dt_list, bins=bins, color='r', alpha=0.5, label='Drift Tube Data')
    plt.title("$p_t$ Histogram")
    plt.xlabel("$p_t$ / $GeVc^{-1}$")
    plt.ylabel("Frequency")
    plt.axvline(x=pt, color='k', linewidth=2,ls='dashed', label='Actual Transverse Momentum')
    plt.legend()
    print 'pt tracker mean' , np.mean(pt_tracker_list)
    print 'pt drift tube mean' , np.mean(pt_dt_list)
    plt.show()

    
#=============================================#
#IMPORTING DATA#

def import_data():
    raw = np.fromfile('C:\Users\Nikos\Documents\GitHub\Muon-Tracking\muon_list.txt', sep=' ')
    raw_split = np.split(raw,40000)
    return raw_split

def pt_calc2(rhit, phihit, zhit,tube):
    x_data, y_data, z_data = smear_data(rhit, phihit, zhit, tube)
    if len(x_data)>0: xc, yc, Rc = leastsq_circle(x_data, y_data)
    if len(x_data)==0: Rc=0
    if tube == 1:
        pt = 0.3*B*Rc
    if tube == 2:
        pt = 0.3*B2*Rc
    if tube == 1:
        k = np.polyfit(r_tracker, z_data,1)[0]
    if tube == 2:
        k = np.polyfit(r_drift, z_data,1)[0]
    pz= pt*k
    return pt, pz
    
    
def pt_datapoint2(pt,pz,phi,charge, iter_num):
    rhit, phihit, zhit = gen_data(pt=pt, pz=pz, phi0=phi, q=charge)
    pt_data,pt_data2,pz_data,pz_data2 = [],[],[],[]
    for i in range(iter_num):
        pt_data = np.append(pt_data, pt_calc2(rhit, phihit, zhit,1)[0])
        pt_data2 = np.append(pt_data2, pt_calc2(rhit, phihit, zhit,2)[0])
        pz_data = np.append(pz_data, pt_calc2(rhit, phihit, zhit,1)[1])
        pz_data2 = np.append(pz_data2, pt_calc2(rhit, phihit, zhit,2)[1])
    phi0 = phihit[0]
    if phihit[1]>phihit[0] : q = -1
    if phihit[1]<phihit[0] : q = 1
    p2 = np.sqrt(pt_data2*pt_data2 + pz_data2*pz_data2)
    p2_mean = np.mean(p2)
    pz_mean = np.mean(pz_data)
    pz_mean2 = np.mean(pz_data2) - (pz_mean/p2_mean)*E_loss #Adjusting for calorimeter energy loss.
    pt_mean = np.mean(pt_data)
    pt_mean2 = np.mean(pt_data2) -(pt_mean/p2_mean)*E_loss #Adjusting for calorimeter energy loss.
    return pt_mean, pt_mean2, pz_mean, pz_mean2, phi0, q
    
def find_fourvector(pt_d,pz_d,phi,charge,iter_num):
    pt, pt2, pz, pz2, phi0, q = pt_datapoint2(pt_d,pz_d,phi,charge,iter_num)
    p_d= np.sqrt(pt_d*pt_d + pz_d*pz_d)
    if p_d <200: 
        p= np.sqrt(pt*pt + pz*pz)
        px = pt*np.cos(phi0)
        py = pt*np.sin(phi0)
        vector = np.array([p, px, py, pz]) 
    if p_d >= 200: 
        p= np.sqrt(pt2*pt2 + pz2*pz2)
        px = pt2*np.cos(phi0)
        py = pt2*np.sin(phi0)
        vector = np.array([p, px, py, pz2])
    return vector
    
def pt_d_calc(data, i):
    pt = np.sqrt((data[i])[1]**2 +(data[i])[2]**2)
    return pt 

def phi0_d_calc(data,i):
    phi0 = np.arctan2((data[i])[2],(data[i])[1])
    return phi0
    
def find_higgs(points, iter_num):
    mass_list, mass_listz1, mass_listz2 = [],[],[]
    data = import_data()
    for i in range(points): 
        muon1 = find_fourvector(pt_d_calc(data,4*i),(data[4*i])[3],phi0_d_calc(data,4*i), (data[4*i])[0], iter_num)
        muon2 = find_fourvector(pt_d_calc(data,4*i+1),(data[4*i+1])[3],phi0_d_calc(data,4*i+1), (data[4*i+1])[0], iter_num)
        muon3 = find_fourvector(pt_d_calc(data,4*i+2),(data[4*i+2])[3],phi0_d_calc(data,4*i+2), (data[4*i+2])[0], iter_num)
        muon4 = find_fourvector(pt_d_calc(data,4*i+3),(data[4*i+3])[3],phi0_d_calc(data,4*i+3), (data[4*i+3])[0], iter_num)
        
        
        z_boson1 = muon1 + muon2
        z_boson2 = muon3 + muon4
        mass_z1 = np.sqrt((z_boson1[0])**2 - ((z_boson1[1])**2 +(z_boson1[2])**2 +(z_boson1[3])**2))
        #print mass_z1
        #print z_boson1
        #print z_boson2
        mass_listz1 = np.append(mass_listz1, mass_z1)
        mass_z2 = np.sqrt((z_boson2[0])**2 - ((z_boson2[1])**2 +(z_boson2[2])**2 +(z_boson2[3])**2))
        #print mass_z2
        mass_listz2 = np.append(mass_listz2, mass_z2)
        
        
        higgs = z_boson1 + z_boson2
        #print higgs
        mass_h = np.sqrt((higgs[0])**2 - ((higgs[1])**2 +(higgs[2])**2 +(higgs[3])**2))
        #print mass_h
        mass_list = np.append(mass_list, mass_h)
            
        #Progress bar.
        sys.stdout.write("\r%d%%" % i)
        sys.stdout.flush()
    mass_mean = np.mean(mass_list)
    mass_mean_z1 = np.mean(mass_listz1)
    mass_mean_z2 = np.mean(mass_listz2)
    return mass_list, mass_listz1, mass_listz2
    
def plot_hist_higgs(points, iter_no):
    h_list, z1_list, z2_list = find_higgs(points, iter_no)
    plt.figure(1)
    bins=np.histogram(z1_list, bins=50)[1]
    plt.hist(z1_list,bins=bins, color='b',normed = False, alpha=1, label='Z Boson')
    #plt.hist(z2_list, color='r',normed = False, alpha=0.5, label='Z Boson 2')
    plt.title("Z Mass Histogram")
    plt.xlabel("$m$ / $GeVc^{-2}$")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.hist(h_list, color='g',normed = False, alpha=1, label='Higgs')
    plt.title("Higgs Mass Histogram")
    plt.xlabel("$m$ / $GeVc^{-2}$")
    plt.ylabel("Frequency")
    plt.axvline(x=126, color='k', linewidth=2,ls='dashed', label='Actual Higgs Mass')
    plt.legend()
    print 'higgs mean' , np.mean(h_list)
    print 'z1 and z2 mean' , np.mean(z1_list), np.mean(z2_list)
    plt.show()
    return h_list, z1_list   
        