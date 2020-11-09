#!/usr/bin/env python3

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError

#local imports
from joints_locator import joints_locator

# Contains methods for calculating the robot's joint's angles given the two camera images
class angles_calculator:

    def __init__(self):
        # initialize the node named angles_calculator
        rospy.init_node('angles_calculator', anonymous=True)
        # initialize a publisher to publish angles to topic named "joints_angles"
        self.angles_pub = rospy.Publisher("joints_angles", Float64MultiArray, queue_size=10)
        # initialize a subscriber to recieve images
        self.image_sub1 = rospy.Subscriber("image_topic1", Image, self.callback1)
        self.image_sub2 = rospy.Subscriber("image_topic2", Image, self.callback2)
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
        yellow = joints_xyz[0,:]
        blue = joints_xyz[1,:]
        green = joints_xyz[2,:]
        red = joints_xyz[3,:]

        # Get vector of link3 (pointing blue->green)
        link3 = green - blue
        # TODO: complete this
        pass




# test the class
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
