"""
nklab.py is a module containing all utility functions of global use in the
project.

Includes:
  1. Functions defined to facilitate coding in other modules.
  2. Plotting functions for the optical system.
"""

# Imports
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function definitions
def normalise(vector):
    """
    Vector is a numpy array.
    """
    unit_vector = vector / np.linalg.norm(vector)
    return unit_vector

def rotate(vector, axis, angle):
    """
    Rotates the unit vector of given vector around given axis by given angle in
    3D space.
    Vector and axis are lists of integers and/or floats, angle is a float
    (in radians).
    """
    vector = normalise(np.array(vector, dtype=float))
    axis = normalise(np.array(axis, dtype=float))
    ax, ay, az = axis[0], axis[1], axis[2]
    cost, sint = np.cos(angle), np.sin(angle)
    R = np.empty((3,3))
    R[0] = np.array([cost+ax*ax*(1-cost), ax*ay*(1-cost)-az*sint,
                     ax*az*(1-cost)+ay*sint])
    R[1] = np.array([ay*ax*(1-cost)+az*sint, cost+ay*ay*(1-cost),
                     ay*az*(1-cost)-ax*sint])
    R[2] = np.array([az*ax*(1-cost)-ay*sint, az*ay*(1-cost)+ax*sint,
                     cost+az*az*(1-cost)])
    rotated_vector = np.dot(R, vector)
    return rotated_vector

def rtpairs(R, N):
    """
    Helps in calling the rtuniform generator
    """
    for i in range(len(R)):
        r = R[i]
        if r == 0:
            t = 0
            yield r, t
        for j in range(N[i]):
            t = j*2*np.pi/N[i]
            yield r, t

def rtuniform(n, m, rmax):
    """
    Parameters given:
      1. n    - number of different radii
              - integer
      2. m    - scale for number of different points on a ring of radius r_n
              - integer
      3. rmax - radius of the disk, r_n less than or equal to rmax
              - float

    Yields:
      1. r - radius of a ring
           - float
      2. t - theta coordinates for a given radius r
           - list

    """
    R = [i*rmax/(1.*n) for i in range(n+1)]
    N = [i*m for i in range(n+1)]
    return rtpairs(R, N)

def plot_rays_2d(rays):
    """
    Plots a bundle of rays on the y-z plane. The rays pass through a refracting
    surface and stop at the output plane.
    rays is a list of rays.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ray in rays:
        y, z = [], []
        for point in ray.vertices():
            y.append(point[1])
            z.append(point[2])
        ax.plot(z, y, color = 'b')
        plt.title('2D plot of collimated rays', fontsize=16)
        plt.xlabel('z', fontsize=14)
        plt.ylabel('y', fontsize=14)


def plot_rays_3d(rays):
    """
    Plots a bundle of rays in 3D space. The rays pass through a refracting
    surface and stop at the output plane.
    n, m, rmax are the parameters of rtuniform.
    """
    fig = plt.figure()
    ax  = Axes3D(fig)

    for ray in rays:
        x, y, z, = [], [], []
        for point in ray.vertices():
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
        ax.plot(x, y, z, color = 'b')
        ax.set_title('3D plot of collimated rays', fontsize=16)
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_zlabel('z', fontsize=14)

def plot_spot_diagram(rays):
    """
    Plots a spot diagram of the xy-positions of the rays at the paraxial focus.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ray in rays:
        ax.plot(ray.r()[0], ray.r()[1], 'bo')
    plt.title('Spot diagram at the paraxial axis', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.gca().set_aspect('equal', adjustable='box')

def plot_initial_spot_diagram(rays):
    """
    Plots a spot diagram of the xy-positions of the rays at z = 0.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ray in rays:
        ax.plot(ray.vertices()[0][0], ray.vertices()[0][1], 'bo')
    plt.title('Spot diagram at z = 0', fontsize=14)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.gca().set_aspect('equal', adjustable='box')

def set_right_curv(curv, focal_length, separation, n1):
    """
    Calculates the curvature of the right surface of a biconvex lens given the
    curvature of the left surface, the separation between the surfaces, the
    refractive index of the lens and the required focal length.
    """
    c = spo.fsolve(lambda c: (n1-1.)*(curv-c + (n1-1.)*curv*c*separation/n1) -
                   1./focal_length, 0.01)[0]
    return c

def diffraction_scale_biconvex(paraxial_focus, lens, wavelength, rmax):
    """
    Calculates the diffraction scale for a biconvex lens.
    """
    diff_scale = (paraxial_focus - lens._z1 + lens._sep/2)*(wavelength*1e-6) /(
                  2*rmax)
    return diff_scale

def disperse(wavelength):
        """
        Calculates a new refractive index n1 for the material depending on the
        wavelength of the incident ray and according to the Sellmeier equation.
        The coefficients used are for a common borosilicate crown glass known
        as BK7.
        """

        wavelength_sq = wavelength*wavelength
        (b1, b2, b3, c1, c2, c3) = (1.03961212, 0.231792344, 1.01046945,
                                    6.00069867e3, 2.00179144e4, 1.03560653e8)

        n_new = np.sqrt(1 + b1*wavelength_sq/(wavelength_sq - c1) +
                            b2*wavelength_sq/(wavelength_sq - c2) +
                            b3*wavelength_sq/(wavelength_sq - c3))
        return n_new

def paraxial_focus(separation, z0, focal_length):
    """
    Calculates the paraxial focus of a lens given its position, separation
    between its two surfaces and the required focal length.
    """
    focus = z0 + separation/2 + focal_length
    return focus

def rms_radius(rays):
    """
    Calculates the rms radius of the given rays at the last point they have
    propagated through.
    """
    r_sqs = []
    for ray in rays:
        x = ray.r()[0]
        y = ray.r()[1]
        r_sqs.append(x*x + y*y)
    rms_radius = np.sqrt(sum(r_sqs)/len(r_sqs))
    return rms_radius

# Exported functions
__all__ = ['normalise', 'rtuniform', 'plot_rays_2d', 'plot_rays_3d',
           'plot_spot_diagram', 'plot_initial_spot_diagram', 'set_right_curv',
           'diffraction_scale_biconvex', 'disperse', 'paraxial_focus',
           'rms_radius']
