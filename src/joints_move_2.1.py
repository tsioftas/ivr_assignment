#!/usr/bin/env python3

import roslib
import sys
import rospy
import numpy as np
from std_msgs.msg import Float64



  # Defines publisher and subscriber
def move_joints():
    # initialize the node named image_processing
    rospy.init_node('joints_move_2_1', anonymous=True)
    
    rate = rospy.Rate(30) # 30hz

    # initialize a publishers to modify joint angles
    joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command",Float64, queue_size=10)
    joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command",Float64, queue_size=10)
    joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command",Float64, queue_size=10)

    # save initialization time
    t0 = rospy.get_time()
    t = 0
    # run for 5.3 seconds
    while t < 5.3:
        t = (np.array([rospy.get_time()]) - t0)
        print(t)
        j2 = (np.pi/2.0) * np.sin(t * np.pi/15)
        j3 = (np.pi/2.0) * np.sin(t * np.pi/18)
        j4 = (np.pi/2.0) * np.sin(t * np.pi/20)

        joint2 = Float64()
        joint2.data = j2
        joint3 = Float64()
        joint3.data = j3
        joint4 = Float64()
        joint4.data = j4


        joint2_pub.publish(joint2)
        joint3_pub.publish(joint3)
        joint4_pub.publish(joint4)
        rate.sleep()
    
    # reset joints to initial position
    zero = Float64()
    zero.data = 0
    joint2_pub.publish(zero)
    joint3_pub.publish(zero)
    joint4_pub.publish(zero)




# run the code if the node is called
if __name__ == '__main__':
  try:
    move_joints()
  except rospy.ROSInterruptException:
    pass


