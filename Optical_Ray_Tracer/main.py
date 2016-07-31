"""
main.py contains the four functions that perform all the final analysis for the
project. The functions define many variables in the start, then perform an
optimisation and eventually plot the results. The command plt.show is needed to
show the plots. The functions included here are:
    1. focal_length_analysis(): produces a graph of varying focal length
    2. wavelength_analysis(): produces a graph of varying wavelength
    3. beam_diameter_analysis(): produces a graph of varying beam diameter
    4. plot_rays(): produces a 2D and a 3D representation of rays passing
       through a biconvex lens as well as their initial and final spot diagram

All variables representing distances are in mm except for the wavelength which
is in nm. The beam diameter is 2*rmax, where rmax is the radius of the
uniformly generated bundle of rays.
The default values producde the graphs of the report.
"""
import ray as tr
import optical_elements as oe
import nklab as nk
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt

reload(tr)
reload(oe)
reload(nk)

def focal_length_analysis(fxi=20., fxf=120., fxs=1.):
    wvlength, z0, n0, separation = 588., 100., 1., 5.
    n1 = nk.disperse(wvlength)
    n, m, rmax = 5, 6, 5.
    curvl, curvr, diff, rms = [], [], [], []
    x0try, maxfuntry = 0.005, 1000
    xvalues = np.arange(fxi, fxf, fxs)
    for focal_length in xvalues:
        paraxial_focus = nk.paraxial_focus(separation, z0, focal_length)
        output = oe.OutputPlane(paraxial_focus)
        def curvature_optimisation_test(curv0):
            curv1 = nk.set_right_curv(curv0, focal_length, separation, n1)
            lensb = oe.BiconvexLens(z0, curv0, curv1, n0, n1, separation)

            rays = []
            for rad, t in nk.rtuniform(n, m, rmax):
                x, y = rad * np.cos(t), rad * np.sin(t)
                ray = tr.Ray(r=[x, y, 0], k=[0, 0, 1], wavelength=wvlength)
                lensb.propagate(ray)
                output.propagate(ray)
                rays.append(ray)

            rms_radius = np.log(nk.rms_radius(rays))

            return rms_radius

        curv0 = spo.fmin_tnc(func = curvature_optimisation_test,
                                        x0 = x0try, approx_grad = True,
                                        maxfun = maxfuntry)[0][0]
        curv1 = nk.set_right_curv(curv0, focal_length, separation, n1)
        initial_guess = [curv0, curv1]

        def curvature_optimisation(curvs):
            curvs = np.array(curvs, dtype=float)
            curv_left, curv_right = curvs[0], curvs[1]
            lensb = oe.BiconvexLens(z0, curv_left, curv_right, n0, n1,
                                    separation)
            rays = []
            for rad, t in nk.rtuniform(n, m, rmax):
                x, y = rad * np.cos(t), rad * np.sin(t)
                ray = tr.Ray(r=[x, y, 0], k=[0, 0, 1], wavelength=wvlength)
                lensb.propagate(ray)
                output.propagate(ray)
                rays.append(ray)

            rms_radius = np.log(nk.rms_radius(rays))
            return rms_radius

        optimal_curv  = spo.fmin_tnc(func = curvature_optimisation,
                 x0 = initial_guess, approx_grad = True, maxfun = maxfuntry)[0]
        curvl.append(optimal_curv[0])
        curvr.append(optimal_curv[1])
        lensb = oe.BiconvexLens(z0, optimal_curv[0], optimal_curv[1],
                                n0, n1, separation)

        rays = []
        for rad, t in nk.rtuniform(n, m, rmax):
            x, y = rad * np.cos(t), rad * np.sin(t)
            ray = tr.Ray(r=[x, y, 0], k=[0, 0, 1], wavelength=wvlength)
            lensb.propagate(ray)
            output.propagate(ray)
            rays.append(ray)

        diff_scale = nk.diffraction_scale_biconvex(paraxial_focus, lensb,
                                                   wvlength, rmax)
        rms_radius = nk.rms_radius(rays)
        diff.append(diff_scale)
        rms.append(rms_radius)

    plt.close("all")
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    diff, rms = np.log(np.array(diff)), np.log(np.array(rms))
    ax1.plot(xvalues, curvl, color = 'c', label='Left surface curvature')
    ax1.plot(xvalues, curvr, color = 'b', label='Right surface curvature')
    ax1.plot(xvalues, [0]*len(xvalues), 'm--')
    ax1.legend(loc='best')

    ax2.plot(xvalues, diff, color = 'g', label='Diffraction scale')
    ax2.plot(xvalues, rms, color = 'r', label='RMS radius')
    ax2.set_xlabel('focal length ('r'mm'')', fontsize=14)
    ax2.legend(loc='best')
    f.subplots_adjust(hspace=0)

def wavelength_analysis(wxi=400., wxf=700., wxs=1.):
    z0, n0, separation = 100., 1., 5.
    n, m, rmax = 5, 6, 5.
    curvl, curvr, diff, rms = [], [], [], []
    focal_length = 75.6
    x0try, maxfuntry = 0.01, 1000
    xvalues = np.arange(wxi, wxf, wxs)

    paraxial_focus = nk.paraxial_focus(separation, z0, focal_length)
    output = oe.OutputPlane(paraxial_focus)
    for wvlength in xvalues:
        n1 = nk.disperse(wvlength)
        def curvature_optimisation_test(curv0):
            curv1 = nk.set_right_curv(curv0, focal_length, separation, n1)
            lensb = oe.BiconvexLens(z0, curv0, curv1, n0, n1, separation)

            rays = []
            for rad, t in nk.rtuniform(n, m, rmax):
                x, y = rad * np.cos(t), rad * np.sin(t)
                ray = tr.Ray(r=[x, y, 0], k=[0, 0, 1], wavelength=wvlength)
                lensb.propagate(ray)
                output.propagate(ray)
                rays.append(ray)

            rms_radius = np.log(nk.rms_radius(rays))

            return rms_radius

        curv0 = spo.fmin_tnc(func = curvature_optimisation_test,
                     x0 = x0try, approx_grad = True, bounds = [(0.017, 0.022)],
                     maxfun = maxfuntry)[0][0]
        curv1 = nk.set_right_curv(curv0, focal_length, separation, n1)
        initial_guess = [curv0, curv1]

        def curvature_optimisation(curvs):
            curvs = np.array(curvs, dtype=float)
            curv_left, curv_right = curvs[0], curvs[1]
            lensb = oe.BiconvexLens(z0, curv_left, curv_right, n0, n1,
                                    separation)

            rays = []
            for rad, t in nk.rtuniform(n, m, rmax):
                x, y = rad * np.cos(t), rad * np.sin(t)
                ray = tr.Ray(r=[x, y, 0], k=[0, 0, 1], wavelength=wvlength)
                lensb.propagate(ray)
                output.propagate(ray)
                rays.append(ray)

            rms_radius = np.log(nk.rms_radius(rays))
            return rms_radius

        optimal_curv = spo.fmin_tnc(func = curvature_optimisation,
                 x0 = initial_guess, approx_grad = True, maxfun = maxfuntry)[0]
        curvl.append(optimal_curv[0])
        curvr.append(optimal_curv[1])
        lensb = oe.BiconvexLens(z0, optimal_curv[0], optimal_curv[1],
                                n0, n1, separation)

        rays = []
        for rad, t in nk.rtuniform(n, m, rmax):
            x, y = rad * np.cos(t), rad * np.sin(t)
            ray = tr.Ray(r=[x, y, 0], k=[0, 0, 1], wavelength=wvlength)
            lensb.propagate(ray)
            output.propagate(ray)
            rays.append(ray)

        diff_scale = nk.diffraction_scale_biconvex(paraxial_focus, lensb,
                                                   wvlength, rmax)
        rms_radius = nk.rms_radius(rays)
        diff.append(diff_scale)
        rms.append(rms_radius)

    plt.close("all")
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    diff, rms = np.log(np.array(diff)), np.log(np.array(rms))
    ax1.plot(xvalues, curvl, 'co', label='Left surface curvature')
    ax1.plot(xvalues, curvr, 'bo', label='Right surface curvature')
    ax1.plot(xvalues, [0]*len(xvalues), 'm--')
    ax1.legend(loc='best')

    ax2.plot(xvalues, diff, 'go', label='Diffraction scale')
    ax2.plot(xvalues, rms, 'ro', label='RMS radius')
    ax2.set_xlabel('wavelength (nm)', fontsize=14)
    ax2.legend(loc='best')
    f.subplots_adjust(hspace=0)

def beam_diameter_analysis(dxi=1., dxf=8., dxs=0.1):
    wvlength, z0, n0, separation = 588., 100., 1., 5.
    n1 = nk.disperse(wvlength)
    n, m = 6, 6
    curvl, curvr, diff, rms = [], [], [], []
    focal_length = 75.6
    x0try, maxfuntry = 0.013, 1000
    xvalues = np.arange(dxi, dxf, dxs)

    paraxial_focus = nk.paraxial_focus(separation, z0, focal_length)
    output = oe.OutputPlane(paraxial_focus)
    for rmax in xvalues:
        def curvature_optimisation_test(curv0):
            curv1 = nk.set_right_curv(curv0, focal_length, separation, n1)
            lensb = oe.BiconvexLens(z0, curv0, curv1, n0, n1, separation)

            rays = []
            for rad, t in nk.rtuniform(n, m, rmax):
                x, y = rad * np.cos(t), rad * np.sin(t)
                ray = tr.Ray(r=[x, y, 0], k=[0, 0, 1], wavelength=wvlength)
                lensb.propagate(ray)
                output.propagate(ray)
                rays.append(ray)

            rms_radius = np.log(nk.rms_radius(rays))

            return rms_radius

        curv0 = spo.fmin_tnc(func = curvature_optimisation_test,
                                        x0 = x0try, approx_grad = True,
                                        maxfun = maxfuntry)[0][0]
        curv1 = nk.set_right_curv(curv0, focal_length, separation, n1)
        initial_guess = [curv0, curv1]

        def curvature_optimisation(curvs):
            curvs = np.array(curvs, dtype=float)
            curv_left, curv_right = curvs[0], curvs[1]
            lensb = oe.BiconvexLens(z0, curv_left, curv_right, n0, n1,
                                    separation)
            rays = []
            for rad, t in nk.rtuniform(n, m, rmax):
                x, y = rad * np.cos(t), rad * np.sin(t)
                ray = tr.Ray(r=[x, y, 0], k=[0, 0, 1], wavelength=wvlength)
                lensb.propagate(ray)
                output.propagate(ray)
                rays.append(ray)

            rms_radius = np.log(nk.rms_radius(rays))
            return rms_radius

        optimal_curv = spo.fmin_tnc(func = curvature_optimisation,
                 x0 = initial_guess, bounds = [(0.01, 0.019), (-0.014, -0.006)],
                 approx_grad = True, maxfun = maxfuntry)[0]
        curvl.append(optimal_curv[0])
        curvr.append(optimal_curv[1])
        lensb = oe.BiconvexLens(z0, optimal_curv[0], optimal_curv[1],
                                n0, n1, separation)

        rays = []
        for rad, t in nk.rtuniform(n, m, rmax):
            x, y = rad * np.cos(t), rad * np.sin(t)
            ray = tr.Ray(r=[x, y, 0], k=[0, 0, 1], wavelength=wvlength)
            lensb.propagate(ray)
            output.propagate(ray)
            rays.append(ray)

        diff_scale = nk.diffraction_scale_biconvex(paraxial_focus, lensb,
                                                   wvlength, rmax)
        rms_radius = nk.rms_radius(rays)
        diff.append(diff_scale)
        rms.append(rms_radius)

    plt.close("all")
    xvalues *=2
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    diff, rms = np.log(np.array(diff)), np.log(np.array(rms))
    ax1.plot(xvalues, curvl, color = 'c', label='Left surface curvature')
    ax1.plot(xvalues, curvr, color = 'b', label='Right surface curvature')
    ax1.plot(xvalues, [0]*len(xvalues), 'm--')
    ax1.legend(loc='center right')

    ax2.plot(xvalues, diff, color = 'g', label='Diffraction scale')
    ax2.plot(xvalues, rms, color = 'r', label='RMS radius')
    ax2.set_xlabel('beam diameter (mm)', fontsize=14)
    ax2.legend(loc='best')
    f.subplots_adjust(hspace=0)

def plot_rays(z0=100., curvl=0.0214922, curvr=-0.004068, n0=1., separation=5.,
              wvlength=588., focal_length=75.6):
    n1 = nk.disperse(wvlength)
    lensb = oe.BiconvexLens(z0, curvl, curvr, n0, n1, separation)
    rays = []
    n, m, rmax = 5, 6, 5.
    paraxial_focus = nk.paraxial_focus(separation, z0, focal_length)
    output = oe.OutputPlane(paraxial_focus)

    for rad, t in nk.rtuniform(n, m, rmax):
        x, y = rad * np.cos(t), rad * np.sin(t)
        ray = tr.Ray(r=[x, y, 0], k=[0, 0, 1], wavelength=wvlength)
        lensb.propagate(ray)
        output.propagate(ray)
        rays.append(ray)
    plt.close("all")
    nk.plot_rays_2d(rays)
    nk.plot_spot_diagram(rays)
    nk.plot_initial_spot_diagram(rays)
    nk.plot_rays_3d(rays)
    plt.show()
