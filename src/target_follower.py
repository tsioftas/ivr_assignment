#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError

#local imports
from joints_locator import joints_locator
import math_utils as mu
import fk

class target_follower:

    def __init__(self):
        # initialize the node
        rospy.init_node('target_follower', anonymous=True)
        self.rate = rospy.Rate(30)
        # initialize subscribers to recieve target and end effector coordinates
        self.targ_x_sub = rospy.Subscriber("target_x_estimate", Float64, self.callback_x)
        self.targ_y_sub = rospy.Subscriber("target_y_estimate", Float64, self.callback_y)
        self.targ_z_sub = rospy.Subscriber("target_z_estimate", Float64, self.callback_z)
        self.ee_pos_sub = rospy.Subscriber("/end_effector_position_estimate", Float64MultiArray, self.callback_ee)
        # Initialize publishers to move joints
        self.joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command",
                                          Float64,
                                          queue_size=10)
        self.joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command",
                                          Float64,
                                          queue_size=10)
        self.joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command",
                                          Float64,
                                          queue_size=10)
        self.joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command",
                                          Float64,
                                          queue_size=10)
        # initialize subscriber to receive robot joint states
        self.joint_states_sub = rospy.Subscriber("/robot/joint_states", JointState, self.callback_joint_states)
        # initialize target coordinates
        self.target_pos = np.array([0, 0, 0.0])
        self.error_prev = np.array([0, 0, 0.0])
        # initialize end effector coordinates
        self.ee_pos = np.array([0, 0, 0.0])
        # initialize joint angles
        self.j = [None, None, None, None]
        # initialize last time a calculation was made
        self.last_t = None


    def callback_x(self, data):
        self.target_pos[0] = data.data

    def callback_y(self, data):
        self.target_pos[1] = data.data
    
    def callback_z(self, data):
        self.target_pos[2] = data.data

    def callback_ee(self, data):
        self.ee_pos = np.array(data.data)

    def callback_joint_states(self, data):
        self.j = np.array(data.position)


    def ready_to_go(self):
        check = [self.target_pos[0],
                 self.target_pos[1],
                 self.target_pos[2],
                 self.error_prev[0],
                 self.error_prev[1],
                 self.error_prev[2],
                 self.j[0],
                 self.j[1],
                 self.j[2],
                 self.j[3],
                 self.last_t,
                 self.ee_pos[0],
                 self.ee_pos[1],
                 self.ee_pos[2]]
        for x in check:
            if x is None:
                return False
        return True
    

    def closed_loop(self):
        # P gain
        K_p = np.array([[1.0, 0, 0],
                        [0.0, 1, 0],
                        [0.0, 0, 1]]) * 5
        # D gain
        K_d = np.array([[1.0, 0, 0],
                        [0.0, 1, 0],
                        [0.0, 0, 1]]) * 0.3
        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            if not self.ready_to_go():
                self.last_t = current_time
                self.error_prev = np.array([0, 0, 0.0])
                self.rate.sleep()
                continue
            dt = current_time - self.last_t
            newerror = self.target_pos - self.ee_pos
            error_dot = np.array(newerror - self.error_prev) / dt
            self.error_prev = newerror
            q = self.j
            j_inv = np.linalg.pinv(fk.calculate_jacobian_matrix(q))

            dq_d = np.matmul(j_inv, (np.matmul(K_d, error_dot.transpose()) + np.matmul(K_p, newerror.transpose())))

            newq = q + dt * dq_d
            self.publish(newq)
            self.last_t = current_time
            self.rate.sleep()

    
    def publish(self, q):
        publishers = [self.joint1_pub, self.joint2_pub, self.joint3_pub, self.joint4_pub]
        for (i,q_i) in enumerate(q):
            to_publish = Float64()
            to_publish.data = q_i
            publishers[i].publish(to_publish)


# create the class
def main():
    tf = target_follower()
    try:
        tf.closed_loop()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main()
