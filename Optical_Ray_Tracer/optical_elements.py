"""
optical_elements.py defines a parent class of optical elements and all daughter
classes which represent the optical elements used to prapagate rays in the
optical system.

Each instantiation is hence an optical element with particular properties,
which are implied by the name (e.g. spherical refracting surface).

The optical axis for all elements is the z-axis and hence all elements are
centred on the z-axis.
"""

import numpy as np
import nklab as nk
import scipy.optimize as spo

class OpticalElement:
    """
    Parent class of all optical elements.
    """
    def propagate(self, ray):
        "propagate a ray through the optical element"
        raise NotImplementedError()


class OutputPlane(OpticalElement):
    """
    Represents the output plane through which rays exit the optical system
    after interacting with the rest of the optical elements.
    """
    def __init__(self, z0):
        """
        Parameter:
          1. z0     - the intercept of the surface with the z-axis
                    - integer or float
        """
        self._z0 = float(z0)

    def __repr__(self):
        return "%s(z0 = %g)" % ("OutputPlane", self._z0)

    def intersect(self, ray):
        """
        Calculates the intercept of a ray with the output plane.
        If the ray is parallel to the plane, it returns None.
        """
        P_ray, k_ray = ray.r(), ray.k()
        if k_ray[2] == 0:
            return None
        else:
            intercept = P_ray + ((self._z0-P_ray[2])/k_ray[2])*k_ray
            return intercept

    def propagate(self, ray):
        """
        Records the intersection point and the direction of the ray when it
        intersects with the output plane.
        """
        if self.intersect(ray) is None:
            raise Exception("Ray does not intersect.")
        intercept = self.intersect(ray)
        direction = ray.k()
        ray.append(intercept, direction)


class CircularRefractingSurface(OpticalElement):
    """
    Represents planar refracting surfaces with circular shape.
    """
    def __init__(self, z0, radius, n0, n1):
        """
        Parameters:
          1. z0        - intercept of the surface with the z-axis
                       - generally float (or integer)
          2. radius    - radius of surface
                       - float or integer
          3. n0        - refractive index outside the surface (usually of air)
                       - generally float (or integer)
          4. n1        - refractive index of the surface
                       - generally float (or integer)
        All parameters are converted into floats.
        """
        self._z0 = float(z0)
        self._radius = float(radius)
        self._n0 = float(n0)
        self._n1 = float(n1)

    def __repr__(self):
        return "%s(z0 = %g, radius = %g, n0 = %g, n1 = %g)" % (
                 "CircularRefractingSurface", self._z0, self._radius,
                 self._n0, self._n1)

    def disperse(self, ray):
        """
        Calculates a new refractive index n1 for the material depending on the
        wavelength of the incident ray and according to the Sellmeier equation.
        The coefficients used are for a common borosilicate crown glass known
        as BK7.
        """
        if ray._wavelength is None:
            return "No wavelength is assigned to the ray"

        wavelength_sq = ray._wavelength*ray._wavelength
        (b1, b2, b3, c1, c2, c3) = (1.03961212, 0.231792344, 1.01046945,
                                    6.00069867e3, 2.00179144e4, 1.03560653e8)

        n_new = np.sqrt(1 + b1*wavelength_sq/(wavelength_sq - c1) +
                            b2*wavelength_sq/(wavelength_sq - c2) +
                            b3*wavelength_sq/(wavelength_sq - c3))
        self._n1 = n_new

    def intersect(self, ray):
        """
        Calculates the valid intercept of a ray with thesurface.
        If there is none, it returns None.
        """
        P_ray, k_ray = ray.r(), ray.k()
        if k_ray[2] == 0:
            return None
        else:
            intercept = P_ray + ((self._z0-P_ray[2])/k_ray[2])*k_ray
            eff_radius = intercept[0]*intercept[0] +  intercept[1]*intercept[1]
            if eff_radius > self._radius*self._radius:
                return None
            return intercept

    def refract(self, ray):
        """
        Calculates the new direction of the ray incident to the refracting
        surface using Snell's law in vector form.
        """
        n = self._n0 / self._n1
        intercept = self.intersect(ray)
        if intercept is None:
            return None
        k1_ray = ray.k()
        normal = -np.array([0., 0., 1]) * (np.sign(k1_ray[2]))
        c = np.dot(-normal, k1_ray)
        sin_t1_sq = 1 - c*c
        if sin_t1_sq > 1./(n*n):
            return None
        k2_ray = nk.normalise(n*k1_ray + (n*c - np.sqrt(1-n*n*(1-c*c)))*normal)
        return k2_ray

    def propagate(self, ray):
        """
        Records the intersection point and the new direction of the ray after
        refraction
        """
        if self.intersect(ray) is not None and self.refract(ray) is not None:
            intercept = self.intersect(ray)
            direction = self.refract(ray)
            ray.append(intercept, direction)


class SphericalRefractingSurface(OpticalElement):
    """
    Represents spherical refracting surfaces.
    """
    def __init__(self, z0, curv, n0, n1, apertureR):
        """
        Parameters:
          1. z0        - intercept of the surface with the z-axis
                       - generally float (or integer)
          2. curv      - curvature of the surface (non-zero)
                       - float
          3. n0        - refractive index outside the surface (usually of air)
                       - generally float (or integer)
          4. n1        - refractive index of the surface
                       - generally float (or integer)
          5. apertureR - aperture radius of surface (less than surface radius)
                       - generally float (or integer)
        All parameters are converted into floats.
        """
        self._z0 = float(z0)
        if curv == 0:
            raise Exception("Curvature cannot be 0. \
                             Instantiate a CircularRefractingSurface instead.")
        self._curv = float(curv)
        self._n0 = float(n0)
        self._n1 = float(n1)
        self._Ra = float(apertureR)

        # __radius is the radius that corresponds to the given curvature and
        # __O is the centre of the spherical surface in Cartesian coordinates.
        self._radius = 1/np.absolute(self._curv)
        if self._radius < self._Ra:
            raise Exception("Aperture radius cannot be more than the radius \
                               of the corresponding sphere.")
        self._O = np.array([0., 0., self._z0 + 1/self._curv])

    def __repr__(self):
        return "%s(z0 = %g, curv = %g, n0 = %g, n1 = %g, apertureR = %g)" % (
                 "SphericalRefractingSurface", self._z0, self._curv,
                 self._n0, self._n1, self._Ra)

    def disperse(self, ray):
        """
        Calculates a new refractive index n1 for the material depending on the
        wavelength of the incident ray and according to the Sellmeier equation.
        The coefficients used are for a common borosilicate crown glass known
        as BK7.
        """
        if ray._wavelength is None:
            return "No wavelength is assigned to the ray"

        wavelength_sq = ray._wavelength*ray._wavelength
        (b1, b2, b3, c1, c2, c3) = (1.03961212, 0.231792344, 1.01046945,
                                    6.00069867e3, 2.00179144e4, 1.03560653e8)

        n_new = np.sqrt(1 + b1*wavelength_sq/(wavelength_sq - c1) +
                            b2*wavelength_sq/(wavelength_sq - c2) +
                            b3*wavelength_sq/(wavelength_sq - c3))
        self._n1 = n_new

    def intersect(self, ray):
        """
        Calculates the valid intercept of a ray with the spherical refracting
        surface if there is one. If there is none, it returns None.
        """

        P_ray, k_ray = ray.r(), ray.k()
        vector_OP = P_ray - self._O

        discriminant = np.dot(vector_OP, k_ray) * np.dot(vector_OP, k_ray) - \
                   (np.dot(vector_OP, vector_OP) - self._radius*self._radius)

        if discriminant < 0:
            return None

        sol1 = -np.dot(vector_OP, k_ray) + np.sqrt(discriminant)
        sol2 = -np.dot(vector_OP, k_ray) - np.sqrt(discriminant)
        intercept1 = P_ray + sol1*k_ray
        intercept2 = P_ray + sol2*k_ray

        if self._curv > 0:
            is_intercept1 = intercept1[2] - self._z0 < self._radius-np.sqrt(
                            self._radius*self._radius - self._Ra*self._Ra)
            is_intercept2 = intercept2[2] - self._z0 < self._radius-np.sqrt(
                            self._radius*self._radius - self._Ra*self._Ra)
        elif self._curv < 0:
            is_intercept1 = intercept1[2] - self._z0 > np.sqrt(self._radius *
                           self._radius - self._Ra*self._Ra) - self._radius
            is_intercept2 = intercept2[2] - self._z0 > np.sqrt(self._radius *
                           self._radius - self._Ra*self._Ra) - self._radius

        if is_intercept1 == False and is_intercept2 == False:
            return None

        is_inside = P_ray[0] > self._O[0] - self._radius and \
                    P_ray[0] < self._O[0] + self._radius and \
                    P_ray[1] > self._O[1] - self._radius and \
                    P_ray[1] < self._O[1] + self._radius and \
                    P_ray[2] > self._O[2] - self._radius and \
                    P_ray[2] < self._O[2] + self._radius

        if (not is_inside and is_intercept1 == True and
                               is_intercept2 == False):
            return intercept1
        elif (not is_inside and is_intercept1 == False and
                                 is_intercept2 == True):
            return intercept2

        if k_ray[2] >= 1e-5 or k_ray[2] <= -1e-5:
            i = 2
        else:
            if k_ray[1] >= 1e-5 or k_ray[1] <= -1e-5:
                i = 1
            else:
                i = 0

        if k_ray[i] > 0:
            if intercept1[i] > intercept2[i]:
                if is_intercept1 == True:
                    return intercept1
                else:
                    return None
            else:
                if is_intercept2 == True:
                    return intercept2
                else:
                    return None
        elif k_ray[i] < 0:
            if intercept1[i] > intercept2[i]:
                if is_intercept2 == True:
                    return intercept2
                else:
                    return None
            else:
                if is_intercept1 == True:
                    return intercept1
                else:
                    return None

    def refract(self, ray):
        """
        Calculates the new direction of the ray incident to the refracting
        surface using Snell's law in vector form.
        """
        n = self._n0 / self._n1
        intercept = self.intersect(ray)
        if intercept is None:
            return None
        k1_ray = ray.k()
        normal = nk.normalise((intercept - self._O) * (np.sign(k1_ray[2]) *
                                           np.sign(self._curv)))
        c = np.dot(-normal, k1_ray)
        sin_t1_sq = 1 - c*c
        if sin_t1_sq > 1./(n*n):
            return None
        k2_ray = nk.normalise(n*k1_ray + (n*c - np.sqrt(1-n*n*(1-c*c)))*normal)
        return k2_ray

    def propagate(self, ray):
        """
        Records the intersection point and the new direction of the ray after
        refraction.
        """
        if self.intersect(ray) is not None and self.refract(ray) is not None:
            intercept = self.intersect(ray)
            direction = self.refract(ray)
            ray.append(intercept, direction)

class PlanoconvexLens(OpticalElement):
    """
    Represents a planoconvex lens.
    """
    def __init__(self, z0, curv, n_out, n_lens, separation):
        """
        Parameters:
          1. z0         - intercept of the spherical surface with the z-axis
                        - generally float (or integer)
          2. curv       - curvature of the spherical surface (non-zero)
                        - float
          3. n_out      - refractive index outside the lens (usually of air)
                        - generally float (or integer)
          4. n_lens     - refractive index of the lens
                        - generally float (or integer)
          5. separation - between planar and spherical surfaces
                        - generally float (or integer)
        All parameters are converted into floats.
        """
        self._z0 = float(z0)
        if curv == 0:
            raise Exception("Curvature of a spherical surface cannot be 0.")
        self._curv = float(curv)
        self._n0 = float(n_out)
        self._n1 = float(n_lens)
        self._sep = float(separation)

        # __Ra is the aperture radius of the lens which depends on the values
        # of curvature and separation.
        self._R = np.absolute(1./self._curv)
        self._Ra = np.sqrt(2*self._R*self._sep +
                            self._sep*self._sep)
        self._z1 = self._z0 + np.sign(self._curv)*self._sep


        if self._curv > 0:
            self.cup = SphericalRefractingSurface(z0=self._z0,
              curv=self._curv, n0=self._n0, n1=self._n1, apertureR=self._Ra)
            self.plane = CircularRefractingSurface(z0=self._z1,
              radius=self._Ra, n0=self._n1, n1=self._n0)
        else:
            self.cup = SphericalRefractingSurface(z0=self._z0,
              curv=self._curv, n0=self._n1, n1=self._n0, apertureR=self._Ra)
            self.plane = CircularRefractingSurface(z0=self._z1,
              radius=self._Ra, n0=self._n0, n1=self._n1)

    def propagate(self, ray):
        """
        Records the intersection point and the new direction of the ray after
        it passes through the lens.
        """
        if self._curv > 0:
            self.cup.propagate(ray)
            self.plane.propagate(ray)
        else:
            self.plane.propagate(ray)
            self.cup.propagate(ray)


class BiconvexLens(OpticalElement):
    """
    Represents a biconvex lens.
    """
    def __init__(self, z0, curv0, curv1, n_out, n_in, separation):
        """
        Parameters:
          1. z0           - intercept of the left surface with the z-axis
                          - generally float (or integer)
          2. curv0, curv1 - curvature of the left and right surfaces
                          - positive and negative float respectively
          3. nout         - refractive index outside the lens (usually of air)
                          - generally float (or integer)
          4. nin          - refractive index of the lens
                          - generally float (or integer)
          5. separation   - between the two surfaces
                          - generally float (or integer)
        All parameters are converted into floats.
        """
        self._z0 = float(z0)
        if curv0 == 0 or curv1 == 0:
            raise Exception("Curvature of a spherical surface cannot be 0.")
        self._curv0 = float(curv0)
        self._curv1 = float(curv1)
        self._n0 = float(n_out)
        self._n1 = float(n_in)
        self._sep = float(separation)

        # __Ra is the aperture radius of the lens which depends on the values
        # of curvature and separation.
        self._R0 = np.absolute(1./self._curv0)
        self._R1 = np.absolute(1./self._curv1)
        self._Ra = self._R0 if self._R0 < self._R1 else self._R1

        self._z1 = self._z0 + self._sep

        self.cup0 = SphericalRefractingSurface(z0=self._z0,
              curv=self._curv0, n0=self._n0, n1=self._n1,
              apertureR=self._Ra)
        self.cup1 = SphericalRefractingSurface(z0=self._z1,
              curv=self._curv1, n0=self._n1, n1=self._n0,
              apertureR=self._Ra)

    def __repr__(self):
        return "%s(z0 = %g, curv0 = %g, curv1 = %g, n_out = %g, n_in = %g, \
                separation = %g)" % ("BiconvexLens", self._z0, self._curv0,
                self._curv1, self._n0, self._n1, self._sep)

    def set_right_curv(self, focal_length):
        """
        Calculates curv1 given curv0.
        """
        c = spo.fsolve(lambda c: (self._n1-1.)*(self._curv0-c + (self._n1-1.)*
            self._curv0*c*self._sep/self._n1) - 1./focal_length, 0.01)[0]
        self._curv1 = c
        return c

    def lensmaker(self):
        """
        Calculates the focus of the lens given its curvature values and
        refractive index according to the lensmaker's equation.
        """
        f_inv = (self._n1-1.)*(self._curv0 - self._curv1 + (self._n1-1.)*
                              self._curv0*self._curv1*self._sep/self._n1)
        f = 1./f_inv
        focus = self._z1 - self._sep/2 + f
        return focus

    def propagate(self, ray):
        """
        Records the intersection point and the new direction of the ray after
        it passes through the lens.
        """
        self.cup0.propagate(ray)
        self.cup1.propagate(ray)

class CircularReflectingSurface(CircularRefractingSurface):
    """
    Represents a planar reflecting surface with circular shape.
    """
    def __init__(self, z0, radius):
        """
        Parameters:
          1. z0        - intercept of the surface with the z-axis
                       - generally float (or integer)
          2. radius    - radius of surface
                       - float or integer
        Both parameters are converted into floats.
        """
        self._z0 = float(z0)
        self._radius = float(radius)

    def refract(self, ray):
        raise NotImplementedError()

    def disperse(self, ray):
        raise NotImplementedError()

    def reflect(self, ray):
        """
        Calculates the reflected direction of the ray incident to the surface.
        """
        intercept = self.intersect(ray)
        if intercept is None:
            return None
        k1_ray = ray.k()
        k2_ray = np.array([k1_ray[0], k1_ray[1], -ray.k()[2]])
        return k2_ray

    def propagate(self, ray):
        """
        Records the intersection point and the new direction of the ray after
        reflection.
        """
        if self.intersect(ray) is None:
            raise Exception("Ray does not intersect.")
        if self.reflect(ray) is None:
            raise Exception("Ray does not reflect.")

        intercept = self.intersect(ray)
        direction = self.reflect(ray)
        ray.append(intercept, direction)

class SphericalReflectingSurface(SphericalRefractingSurface):
    """
    Represents a spherical reflecting surface
    """
    def __init__(self, z0, curv, apertureR):
        """
        Parameters:
          1. z0        - intercept of the surface with the z-axis
                       - generally float (or integer)
          2. curv      - curvature of the surface (non-zero)
                       - float
          5. apertureR - aperture radius of surface (less than surface radius)
                       - generally float (or integer)
        All parameters are converted into floats.
        """
        self._z0 = float(z0)
        if curv == 0:
            raise Exception("Curvature cannot be 0. \
                             Instantiate a CircularRefractingSurface instead.")
        self._curv = float(curv)
        self._Ra = float(apertureR)

        # __radius is the radius that corresponds to the given curvature and
        # __O is the centre of the spherical surface in Cartesian coordinates.
        self._radius = 1/np.absolute(self._curv)
        if self._radius < self._Ra:
            raise Exception("Aperture radius cannot be more than the radius \
                               of the corresponding sphere.")
        self._O = np.array([0., 0., self._z0 + 1/self._curv])

    def refract(self, ray):
        raise NotImplementedError()

    def disperse(self, ray):
        raise NotImplementedError()

    def reflect(self, ray):
        """
        Calculates the reflected direction of the ray incident to the surface.
        """
        intercept = self.intersect(ray)
        if intercept is None:
            return None
        k1_ray = ray.k()
        normal = nk.normalise((intercept - self._O) * (np.sign(k1_ray[2]) *
                                           np.sign(self._curv)))
        cos_t = - np.dot(normal, k1_ray)
        k2_ray = k1_ray + 2*cos_t*normal
        return k2_ray

    def propagate(self, ray):
        """
        Records the intersection point and the new direction of the ray after
        reflection.
        """
        if self.intersect(ray) is None:
            raise Exception("Ray does not intersect.")
        if self.reflect(ray) is None:
            raise Exception("Ray does not reflect.")

        intercept = self.intersect(ray)
        direction = self.reflect(ray)
        ray.append(intercept, direction)
