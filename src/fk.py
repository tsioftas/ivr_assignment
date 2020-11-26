#!/usr/bin/env python3

import numpy as np

# local imorts
import constants

def calculate_fk_matrix(q):
    c1 = np.cos(q[0])
    c2 = np.cos(q[1])
    c3 = np.cos(q[2])
    c4 = np.cos(q[3])
    s1 = np.sin(q[0])
    s2 = np.sin(q[1])
    s3 = np.sin(q[2])
    s4 = np.sin(q[3])

    l1 = constants.get_link_length(1)
    #l2 = -constants.get_link_length(2)
    l3 = -constants.get_link_length(3)
    l4 = -constants.get_link_length(4)

    m00 = -s1*s2*c3*c4 - c1*s3*c4 - s1*c2*s4
    m01 = s1*s2*c3*s4 + c1*s3*s4 - s1*c2*c4
    m02 = -s1*s2*s3 + c1*c3
    m03 = l4*(-s1*s2*c3*c4 - c1*s3*c4 - s1*c2*s4) - l3*(s1*s2*c3 + c1*s3)

    m10 = c1*s2*c3*c4 - s1*s3*c4 + c1*c2*s4
    m11 = -c1*s2*c3*s4 + s1*s3*s4 + c1*c2*c4
    m12 = c1*s2*s3 + s1*c3
    m13 = l4*(c1*s2*c3*c4 - s1*s3*c4 + c1*c2*s4) + l3*(c1*s2*c3 - s1*s3)

    m20 = -c2*c3*c4 + s2*s4
    m21 = c2*c3*s4 + s2*c4
    m22 = -c2*s3
    m23 = l4*(-c2*c3*c4 + s2*s4) - l3*c2*c3 + l1

    matrix = np.array([[m00, m01, m02, m03],
                       [m10, m11, m12, m13],
                       [m20, m21, m22, m23],
                       [0, 0, 0, 1]])
    return matrix


def get_end_effector_xyz(q):
    matrix = calculate_fk_matrix(q)
    return matrix[0:3, 3]

# test the functions
if __name__ == "__main__":
    theta1 = -1.154
    theta2 = 0.465
    theta3 = -1.537
    theta4 = 0.845
    test_q = np.array([theta1, theta2, theta3, theta4])
    print(get_end_effector_xyz(test_q))
