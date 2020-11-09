#!/usr/bin/env python3

import numpy as np

def is_3d_vector(v):
    return v.shape == (1, 3) or v.shape == (3, 1)

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
