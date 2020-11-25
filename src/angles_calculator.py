#!/usr/bin/env python3

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError

#local imports
from joints_locator import joints_locator
import math_utils as mu

# Contains methods for calculating the robot's joint's angles given the two camera images
class angles_calculator:

    def __init__(self):
        # initialize the node named angles_calculator
        rospy.init_node('angles_calculator', anonymous=True)
        # initialize a publisher to publish angles to topic named "joints_angles"
        self.a1_pub = rospy.Publisher("joint2_angle_estimate", Float64, queue_size=10)
        self.a2_pub = rospy.Publisher("joint3_angle_estimate", Float64, queue_size=10)
        self.a3_pub = rospy.Publisher("joint4_angle_estimate", Float64, queue_size=10)
        # initialize a subscriber to recieve images
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # initialize images
        self.image1 = None
        self.image2 = None
        # initialize images' updated status
        self.updated1 = False
        self.updated2 = False
        # initialize joints locator
        self.jl = joints_locator()

    def callback1(self, data):
        # Save the received image:
        try:
            self.image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.updated1 = True
        except CvBridgeError as e:
            self.updated1 = False
            print(e)
        # Only call if both images have been received
        if(self.updated1 and self.updated2):
            self.updated1 = False
            self.updated2 = False
            self.calculate_angles()

    def callback2(self, data):
        # Save the received image:
        try:
            self.image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.updated2 = True
        except CvBridgeError as e:
            self.updated2 = False
            print(e)
        # Only call if both images have been received
        if(self.updated1 and self.updated2):
            self.updated1 = False
            self.updated2 = False
            self.calculate_angles()

    def calculate_angles(self):
        joints_xyz = self.jl.get_joints_xyz_locations(self.image1, self.image2)
        # position vectors of each joint:
        yellow = joints_xyz[0, :]
        blue = joints_xyz[1, :]
        green = joints_xyz[2, :]
        red = joints_xyz[3, :]

        # Get vector of link3 (pointing blue->green)
        link3 = green - blue
        print(link3)

        # calculate a1
        # find projection of link3 onto yz plane
        n_yz = np.array([1, 0, 0]) # normal to yz plane
        link3_yz = mu.project_plane(link3, n_yz)
        a1 = mu.angle_between_vector_and_plane(link3_yz, np.array([0,-1,0])) # angle formed with xz plane, positive direction is towards negative y

        # calculate a2
        a2 = mu.angle_between_vector_and_plane(link3, n_yz) # angle formed with yz plane, positive direction is towards positive x

        n3 = np.array([0,-1,0]) # normal to the plane of rotation of 4th joint at initial state
        x_axis = np.array([1,0,0])
        y_axis = np.array([0,1,0])
        n3 = mu.rotation_sequence([x_axis, y_axis, n3], [a1, a2])
        link4 = red - green
        a3 = mu.angle_between_vector_and_plane(link4, n3)

        joint2 = Float64()
        joint2.data = a1
        joint3 = Float64()
        joint3.data = a2
        joint4 = Float64()
        joint4.data = a3

        self.a1_pub.publish(joint2)
        self.a2_pub.publish(joint3)
        self.a3_pub.publish(joint4)
        

# call the class
def main(args):
    ac = angles_calculator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
