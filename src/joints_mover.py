#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64

class joints_mover:
    """Class for moving joints"""

    def __init__(self):
        # initialize the node
        rospy.init_node('joints_mover', anonymous=True)
        self.rate = rospy.Rate(30) # 30hz
        # initialize publishers to modify joint angles
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


    def move_joints_2_1(self, delta_t=5.3):
        """Function called when the node is run to move the joints
        according to the specifications of 2.1"""
        # save initialization time
        t0 = rospy.get_time()
        t = 0
        # run for delta_t seconds
        while t < delta_t:
            t = (np.array([rospy.get_time()]) - t0)
            print(t)
            # Calculate angles
            j2 = (np.pi/2.0) * np.sin(t * np.pi/15)
            j3 = (np.pi/2.0) * np.sin(t * np.pi/18)
            j4 = (np.pi/2.0) * np.sin(t * np.pi/20)
            self.set_joint_angle(2, j2)
            self.set_joint_angle(3, j3)
            self.set_joint_angle(4, j4)
            self.rate.sleep()
        self.reset_joints()


    # Sets the angle of the specified joint (1-4) to the given value
    def set_joint_angle(self, joint, angle):
        publishers = [self.joint1_pub, self.joint2_pub, self.joint3_pub, self.joint4_pub]
        to_publish = Float64()
        to_publish.data = angle
        publishers[joint].publish(to_publish)


    # Sets all joint angles to zero
    def reset_joints(self):
        for i in range(1, 5):
            self.set_joint_angle(i, 0)




# run the code if the node is called
if __name__ == '__main__':
    try:
        jm = joints_mover()
        jm.move_joints_2_1()
    except rospy.ROSInterruptException:
        pass
