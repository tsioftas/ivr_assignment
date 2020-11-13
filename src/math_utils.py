#!/usr/bin/env python3

import numpy as np

def is_3d_vector(v):
    return v.shape == (1, 3) or v.shape == (3, 1) or v.shape == (3, )

# returns the projection of vector u onto vector onto_v
def project_vector(u, onto_v):
    assert is_3d_vector(u) and is_3d_vector(onto_v)
    factor = np.dot(u, onto_v) / np.dot(onto_v, onto_v)
    return factor * onto_v

# returns the projection of vector u onto plane whose normal is vector onto_n
def project_plane(u, onto_n):
    assert is_3d_vector(u) and is_3d_vector(onto_n)
    proj = project_vector(u, onto_n)
    return u-proj

# returns the convex angle between vectors u and v
def angle_between_vectors(u, v):
    assert is_3d_vector(u) and is_3d_vector(v)
    costheta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.arccos(costheta)

# returns the convex angle between vector u and plane whose normal is vector n
def angle_between_vector_and_plane(u, n):
    assert is_3d_vector(u) and is_3d_vector(n)
    return np.pi/2 - angle_between_vectors(u, n)

# returns the resulting vector after rotating v around axis u by t radians
# u must be a unit vector
def rotate_vector_around_axis_by_t(v, u, t):
    assert is_3d_vector(v) and is_3d_vector(u)
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

# Given a sequence of axes and angles, rotates all subsequent axes by each rotation axis
# so that the final vector is the result of all the rotations TODO: rewrite this comment
def rotation_sequence(axes, angles):
    for i in range(len(axes)-1):
        for j in range(i+1, len(axes)):
            # rotate axis j around axis i by angle angles[i]
            axes[j] = rotate_vector_around_axis_by_t(axes[j], axes[i], angles[i])
    return axes[-1]

