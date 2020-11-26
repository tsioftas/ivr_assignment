#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# local imports
from joints_locator import joints_locator

class end_effector_tracker:
    """Used to track the xyz position of the end effector using the view from the two cameras"""

    def __init__(self):
        # initialize the node
        rospy.init_node('end_effector_tracker', anonymous=True)
        self.rate = rospy.Rate(30) # 30hz
        # initialize subscribers to receive images
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.record_img1)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.record_img2)
        # initialize publishers to publish the estimate
        self.ee_pos_pub = rospy.Publisher("/end_effector_position_estimate", Float64MultiArray, queue_size=10)
        # initialize images
        self.img1 = None
        self.img2 = None
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # initialize joints locator
        self.jl = joints_locator()

    # Callback functions for saving the images
    def record_img1(self, data):
        try:
            self.img1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def record_img2(self, data):
        try:
            self.img2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.track()


    # Track the position of the target. Runs for a default of 10.5 seconds.
    # Parameter delta_t can be used to choose a different time period. Parameter
    # forever can be set to True to track the target until interrupted.
    def track(self, forever=False, delta_t=10.5):
        if (self.img1 is not None) and (self.img2 is not None):
            coords = self.jl.get_joints_xyz_locations_meters(self.img1, self.img2)[-1,:]
            pub = Float64MultiArray()
            pub.data = [coords[0], coords[1], coords[2]]
            print(coords)
            self.ee_pos_pub.publish(pub)


# run the code if the node is called
if __name__ == '__main__':
    try:
        eet = end_effector_tracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
