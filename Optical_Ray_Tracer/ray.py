"""
ray.py defines a class of rays that can be represented in space. A ray
propagates in the optical system and can be refracted, reflected or dispersed.

Each instantiation is hence described by several line segments in space which
are determined by their endpoints and directions. The final segment determines
the current direction of the ray.
"""

import numpy as np
import nklab as nk

class Ray:
    """
    Instantiates an optical ray.

    Provides
      1. A vector representation of the ray in the system.
      2. Methods for updating the representation of the ray and returning its
         current point and direction each time it propagates to an optical
         element surface.
    """
    def __init__(self, r=[0, 0, 0], k=[0, 0, 1], wavelength = 0):
        """
        Instantiates an optical ray at a starting position r with initial
        (normalised) direction k. Coordinates are in the x,y,z Cartesian form.

        r and k can be numpy arrays or lists of integers and/or floats.
        wavelength is a float (measured in nanometres).
        """
        if len(r) != 3 or len(k) != 3:
            raise Exception('3D vector size')

        self._r = np.array(r, dtype=float)
        self._k = nk.normalise(np.array(k, dtype=float))
        if wavelength == 0:
            self._wavelength = None
        self._wavelength = float(wavelength)

        # __vertices and __directions are lists of all segment endpoints and
        # directions of the ray. They are useful for plotting but not useful
        # for the user.
        self._vertices = [self._r]
        self._directions = [self._k]

    def __repr__(self):
        """
        Represents the current point and direction of the ray
        """
        return "%s(r=[%g, %g, %g], k=[%g, %g, %g])" % (
                "Ray", self.r()[0], self.r()[1], self.r()[2],
                self.k()[0], self.k()[1], self.k()[2])

    def __str__(self):
        """
        Represents the current point and direction of the ray
        """
        return "r = (%g, %g, %g), k = (%g, %g, %g)" % (
                self.r()[0], self.r()[1], self.r()[2],
                self.k()[0], self.k()[1], self.k()[2])

    def r(self):
        """
        Gets the value of the current point.
        """
        return self._vertices[-1]

    def k(self):
        """
        Gets the value of the current direction.
        """
        return self._directions[-1]

    def vertices(self):
        """
        Gets the values of all vertices of the ray.
        Vertices are numpy arrays of floats.
        """
        return self._vertices

    def append(self, r, k):
        """
        Appends new point and direction to the ray usually after interaction
        with optical element.

        r, k can be numpy arrays or lists of floats and/or integers.
        Appended points and directions are numpy arrays of floats.
        Directions are normalised.
        """
        if len(r) != 3 or len(k) != 3:
            raise Exception('3D vector size')

        r = np.array(r, dtype=float)
        k = nk.normalise(np.array(k, dtype=float))

        self._vertices.append(r)
        self._directions.append(k)
