#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError

#local imports
from joints_locator import joints_locator
import math_utils as mu

class angles_estimator:
    """
    Subscribes to images from the cameras and uses them to estimate the joint angles.
    The estimates are published in topics \"joint2_angle_estimate\", \"joint3_angle_estimate\" and
    \"joint4_angle_estimate\" each of them as a Float64.
    """


    def __init__(self):
        # initialize the node
        rospy.init_node('angles_estimator', anonymous=True)
        # initialize publishers to publish estimated angles
        self.j2_pub = rospy.Publisher("joint2_angle_estimate", Float64, queue_size=10)
        self.j3_pub = rospy.Publisher("joint3_angle_estimate", Float64, queue_size=10)
        self.j4_pub = rospy.Publisher("joint4_angle_estimate", Float64, queue_size=10)
        # initialize subscribers to recieve images
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
        self.calculate_angles_safe()

    def callback2(self, data):
        # Save the received image:
        try:
            self.image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.updated2 = True
        except CvBridgeError as e:
            self.updated2 = False
            print(e)
        self.calculate_angles_safe()


    # This method ensures calculate_angles is only called once both images have been received
    def calculate_angles_safe(self):
        # Only call if both images have been received
        if(self.updated1 and self.updated2):
            self.updated1 = False
            self.updated2 = False
            self.calculate_angles()


    # This method uses the joints locator object to extract the xyz coordinates of each joint
    # from the images. It then uses some linear algebra and the math_utils module to calculate
    # the joints angles from their coordinates.
    def calculate_angles(self):
        joints_xyz = self.jl.get_joints_xyz_locations(self.image1, self.image2)
        # Position vectors of each joint. Origin is at the yellow joint, axes orientation
        # is the same as in figure1 of the specifications document.
        blue = joints_xyz[1, :]
        green = joints_xyz[2, :]
        red = joints_xyz[3, :]

        # Get vector of link3 (pointing blue->green)
        link3 = green - blue

        # Find projection of link3 onto yz plane
        n_yz = np.array([1, 0, 0]) # normal to yz plane
        link3_yz = mu.project_plane(link3, n_yz)
        # joint2 is the angle link3_yz forms with xz plane, positive direction is towards negative y
        n_xz = np.array([0, -1, 0])
        j2 = mu.angle_between_vector_and_plane(link3_yz, n_xz)

        # joint3 is the angle link3 forms with yz plane, positive direction is towards positive x
        j3 = mu.angle_between_vector_and_plane(link3, n_yz) #

        n3 = np.array([0, -1, 0]) # Normal to the plane of rotation of 4th joint at initial state
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        # Find transformed n3 after rotations of joint2 and joint3
        n3 = mu.rotation_sequence([x_axis, y_axis, n3], [j2, j3])
        # Get vector for link4 (pointing green->red)
        link4 = red - green
        # joint4 is the angle link4 forms with the joint's transformed xz plane
        j4 = mu.angle_between_vector_and_plane(link4, n3)

        # Publish the estimates
        joint2 = Float64()
        joint2.data = j2
        joint3 = Float64()
        joint3.data = j3
        joint4 = Float64()
        joint4.data = j4
        self.j2_pub.publish(joint2)
        self.j3_pub.publish(joint3)
        self.j4_pub.publish(joint4)


# create the class
def main():
    _ = angles_estimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main()
