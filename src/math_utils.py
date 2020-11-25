#!/usr/bin/env python3

import numpy as np

# Tests whether the given argument has the correct shape to be a 3d vector
def is_3d_vector(v):
    return v.shape == (1, 3) or v.shape == (3, 1) or v.shape == (3, )

# Returns the projection of vector u onto vector onto_v
def project_vector(u, onto_v):
    assert is_3d_vector(u) and is_3d_vector(onto_v)
    factor = np.dot(u, onto_v) / np.dot(onto_v, onto_v)
    return factor * onto_v

# Returns the projection of vector u onto plane whose normal is vector onto_n
def project_plane(u, onto_n):
    assert is_3d_vector(u) and is_3d_vector(onto_n)
    proj = project_vector(u, onto_n)
    return u-proj

# Returns the convex angle between vectors u and v
def angle_between_vectors(u, v):
    assert is_3d_vector(u) and is_3d_vector(v)
    costheta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.arccos(costheta)

# Returns the convex angle between vector u and plane whose normal is vector n
def angle_between_vector_and_plane(u, n):
    assert is_3d_vector(u) and is_3d_vector(n)
    return np.pi/2 - angle_between_vectors(u, n)

# Returns the resulting vector after rotating v around axis u by t radians.
def rotate_vector_around_axis_by_t(v, u, t):
    assert is_3d_vector(v) and is_3d_vector(u)
    normu = np.linalg.norm(u)
    if not eq(normu, 1.0):
        # u must be a unit vector. Normalize.
        u /= normu
    # Construct rotation matrix from axis and angle.
    # See here: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    ct = np.cos(t)
    st = np.sin(t)
    ux = u[0]
    uy = u[1]
    uz = u[2]
    r1 = np.array([ct+ux*ux*(1-ct), ux*uy*(1-ct)-uz*st, ux*uz*(1-ct)+uy*st])
    r2 = np.array([uy*ux*(1-ct)+uz*st, ct+uy*uy*(1-ct), uy*uz*(1-ct)-ux*st])
    r3 = np.array([uz*ux*(1-ct)-uy*st, uz*uy*(1-ct)+ux*st, ct+uz*uz*(1-ct)])
    matrix = np.array([r1, r2, r3])
    return np.matmul(matrix, v)

# Given a sequence of axes and angles, for each axes[i], angles[i], sequentially rotates
# all the following axes[j] (j>i) around axes[i] by angles[i]. Returns the eventual result of
# for the last axis.
def rotation_sequence(axes, angles):
    for i in range(len(axes)-1):
        for j in range(i+1, len(axes)):
            # rotate axis j around axis i by angle angles[i]
            axes[j] = rotate_vector_around_axis_by_t(axes[j], axes[i], angles[i])
    return axes[-1]

# Tests two floating point numbers for equality
def eq(a, b, epsilon=1e-9):
    return abs(a-b) < epsilon
