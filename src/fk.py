#!/usr/bin/env python3

import sys
import numpy as np
from std_msgs.msg import Float64
import rospy

# local imorts
import constants
import math_utils as mu

def frame_curr_to_prev(alpha, a, d, theta):
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)
    matrix = np.array([[ct, -st*ca, st*sa, a*ct],
                       [st, ct*ca, -ct*sa, a*st],
                       [0, sa, ca, d],
                      [0, 0, 0, 1]])
    return matrix

def fk_cheat(q):
    l1 = constants.get_link_length(1)
    #l2 = -constants.get_link_length(2)
    l3 = constants.get_link_length(3)
    l4 = constants.get_link_length(4)
    alphas = [np.pi/2, -np.pi/2, np.pi/2, 0]
    a_s = [0, 0, -l3, -l4]
    ds = [l1, 0, 0, 0]
    thetas = [q[0]+np.pi/2, q[1]-np.pi/2, q[2], q[3]]
    res = np.eye(4)
    for i in range(4):
        res = np.matmul(res, frame_curr_to_prev(alphas[i], a_s[i], ds[i], thetas[i]))
    return res[0:3, 3]

def calculate_fk_matrix(qq):
    cs = np.array([np.pi/2, -np.pi/2, 0, 0])
    # Add constants
    q = qq + cs

    c1 = np.cos(q[0])
    c2 = np.cos(q[1])
    c3 = np.cos(q[2])
    c4 = np.cos(q[3])
    s1 = np.sin(q[0])
    s2 = np.sin(q[1])
    s3 = np.sin(q[2])
    s4 = np.sin(q[3])

    l1 = constants.get_link_length(1)
    #l2 = constants.get_link_length(2)
    l3 = constants.get_link_length(3)
    l4 = constants.get_link_length(4)

    m00 = c1*c2*c3*c4 - s1*s3*c4 - c1*s2*s4
    m01 = -c1*c2*c3*s4 + s1*s3*s4 - c1*s2*c4
    m02 = c1*c2*s3 + s1*c3
    m03 = -l4*(c1*c2*c3*c4 - s1*s3*c4 - c1*s2*s4) - l3*(c1*c2*c3 - s1*s3)

    m10 = s1*c2*c3*c4 + c1*s3*c4 - s1*s2*s4
    m11 = -s1*c2*c3*s4 - c1*s3*s4 - s1*s2*c4
    m12 = s1*c2*s3 - c1*s3
    m13 = -l4*(s1*c2*c3*c4 + c1*s3*c4 - s1*s2*s4) - l3*(s1*c2*c3 + c1*s3)

    m20 = s2*c3*c4 + c2*s4
    m21 = -s2*c3*s4 + c2*c4
    m22 = s2*s3 - c2*c3
    m23 = -l4*(s2*c3*c4 + c2*s4) - l3*(s2*c3) + l1

    matrix = np.array([[m00, m01, m02, m03],
                       [m10, m11, m12, m13],
                       [m20, m21, m22, m23],
                       [0, 0, 0, 1]])
    return matrix


def  calculate_jacobian_matrix(q):
    c1 = np.cos(q[0])
    c2 = np.cos(q[1])
    c3 = np.cos(q[2])
    c4 = np.cos(q[3])
    s1 = np.sin(q[0])
    s2 = np.sin(q[1])
    s3 = np.sin(q[2])
    s4 = np.sin(q[3])
    l1 = constants.get_link_length(1)
    #l2 = constants.get_link_length(2)
    l3 = constants.get_link_length(3)
    l4 = constants.get_link_length(4)

    j00 = l4*(c1*s2*c3*c4 - s1*s3*c4 + c1*c2*s4) + l3*(c1*s2*c3 - s1*s3)
    j01 = l4*(s1*c2*c3*c4 - s1*s2*s4) + l3*s1*c2*c3
    j02 = l4*(-s1*s2*s3*c4 + c1*c3*c4) + l3*(-s1*s2*s3 + c1*c3)
    j03 = l4*(-s1*s2*c3*s4 - c1*s3*s4 + s1*c2*c4)

    j10 = l4*(s1*s2*c3*c4 + c1*s3*c4 + s1*c2*s4) + l3*(s1*s2*c3 + c1*s3)
    j11 = l4*(-c1*c2*c3*c4 + c1*s2*s4) + l3*(-c1*c2*c3)
    j12 = l4*(c1*s2*s3*c4 + s1*c3*c4) + l3*(c1*s2*s3 + s1*c3)
    j13 = l4*(c1*s2*c3*s4 - s1*s3*s4 - c1*c2*c4)

    j20 = 0
    j21 = l4*(-s2*c3*c4 - c2*s4) - l3*s2*c3
    j22 = l4*(-c2*s3*c4) - l3*c2*s3
    j23 = l4*(-c2*c3*s4 - s2*c4)

    jacobian = np.array([[j00, j01, j02, j03],
                         [j10, j11, j12, j13],
                         [j20, j21, j22, j23]])
    return jacobian



def get_green_joint_xyz(qq):
    cs = np.array([np.pi/2, -np.pi/2, 0, 0])
    # Add constants
    q = qq + cs
    l1 = constants.get_link_length(1)
    l3 = constants.get_link_length(3)
    c1 = np.cos(q[0])
    c2 = np.cos(q[1])
    c3 = np.cos(q[2])
    s1 = np.sin(q[0])
    s2 = np.sin(q[1])
    s3 = np.sin(q[2])
    # Calculated from intermediate step in calculating FK matrix
    x = l3*(c1*c2*c3 - s1*s3)
    y = l3*(s1*c2*c3 + c1*s3)
    z = l3*(s2*c3) + l1
    return np.array([x, y, z])


def get_end_effector_xyz(q):
    matrix = calculate_fk_matrix(q)
    return matrix[0:3, 3]


def get_random_angles(n_tests=10):
    ret = np.zeros((n_tests, 4))
    for i in range(n_tests):
        has_zero = True
        angles = None
        while has_zero:
            has_zero = False
            angles = np.random.rand(4)*np.pi - np.pi/2
            for angle in angles:
                has_zero |= mu.eq(angle, 0, epsilon=1e-2) # loose zero testing
        ret[i,:] = angles
    return ret

# ---Random angles for testing---
angles = np.array([[-1.11660767, -0.60247749,  0.89124467, -0.93814622],
                    [ 0.41813778,  1.48922339, -0.24340268, 0.01884399],
                    [ 1.56354762, -0.50085079,  1.56450093,  0.68618036],
                    [-1.08668677, -0.83718411,  0.21872362, -1.51275805],
                    [ 1.34600929,  0.61314554,  1.50190585,  0.13056269],
                    [ 0.14122202,  1.38556733, -0.43694138,  1.25202932],
                    [-1.28243129, -1.53239745, -0.6816539,  -0.17402252],
                    [ 0.94992629, -0.66843285,  0.21449914, -0.26452517],
                    [-0.84842083,  1.31555105,  1.00870696,  0.19971582],
                    [-1.13493569, -0.28506399, -1.20625613,  1.44617784]])
 #-------------------

def run_test(i):
    rospy.init_node('fk_tester', anonymous=True)
    joint_pubs = []
    joint_pubs.append(rospy.Publisher("/robot/joint1_position_controller/command",
                                        Float64,
                                        queue_size=10))
    joint_pubs.append(rospy.Publisher("/robot/joint2_position_controller/command",
                                        Float64,
                                        queue_size=10))
    joint_pubs.append(rospy.Publisher("/robot/joint3_position_controller/command",
                                        Float64,
                                        queue_size=10))
    joint_pubs.append(rospy.Publisher("/robot/joint4_position_controller/command",
                                        Float64,
                                        queue_size=10))
    test_q = angles[i,:]
    print("ANGLES: {}".format(test_q))
    fk_pos = get_end_effector_xyz(test_q.copy())
    print(fk_pos)
    while not rospy.is_shutdown():
        for (j,theta) in enumerate(test_q):
            joint_pubs[j].publish(theta)


if __name__ == "__main__":
    try:
        choice = int(sys.argv[1])
        if 0 <= choice and choice  <= 9:
            run_test(choice)
    except rospy.ROSInterruptException:
        pass

    # theta1 = 1.2
    # theta2 = 0.8
    # theta3 = 1
    # theta4 = 0.2
    # test_q = np.array([theta1, theta2, theta3, theta4])
    # print(fk_cheat(test_q))
    # print(get_end_effector_xyz(test_q))

