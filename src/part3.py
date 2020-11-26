#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import math
from random import randrange
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import joints_locator


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send messages to a topic named image_topic
    self.image_pub = rospy.Publisher("image_topic",Image, queue_size = 1)
    # initialize a publisher to send joints' angular position to a topic called joints_pos
    self.joints_pub = rospy.Publisher("joints_pos",Float64MultiArray, queue_size=10)
    # initialize a publisher to send robot end-effector position
    self.end_effector_pub = rospy.Publisher("end_effector_prediction",Float64MultiArray, queue_size=10)
    # initialize a publisher to send desired trajectory
    self.trajectory_pub = rospy.Publisher("trajectory",Float64MultiArray, queue_size=10)
    # initialize a publisher to send joints' angular position to the robot
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub = rospy.Subscriber("/robot/camera1/image_raw",Image,self.callback)
    # record the begining time
    self.time_trajectory = rospy.get_time()
    # initialize errors
    self.time_previous_step = np.array([rospy.get_time()], dtype='float64')     
    self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')   
    # initialize error and derivative of error for trajectory tracking  
    self.error = np.array([0.0,0.0], dtype='float64')  
    self.error_d = np.array([0.0,0.0], dtype='float64') 



  # Calculate the conversion from pixel to meter
  def pixel2meter(self,image):
      # Obtain the centre of each coloured blob
      circle1Pos = self.detect_blue(image)
      circle2Pos = self.detect_green(image)
      # find the distance between two circles
      dist = np.sum((circle1Pos - circle2Pos)**2)
      return 3 / np.sqrt(dist)

    # Calculate the relevant joint angles from the image
  def detect_joint_angles(self,image):
    a = self.pixel2meter(image)
    # Obtain the centre of each coloured blob 
    center = a * self.detect_yellow(image)
    circle1Pos = a * self.detect_blue(image) 
    circle2Pos = a * self.detect_green(image) 
    circle3Pos = a * self.detect_red(image)
    # Solve using trigonometry
    ja1 = np.arctan2(center[0]- circle1Pos[0], center[1] - circle1Pos[1])
    ja2 = np.arctan2(circle1Pos[0]-circle2Pos[0], circle1Pos[1]-circle2Pos[1]) - ja1
    ja3 = np.arctan2(circle2Pos[0]-circle3Pos[0], circle2Pos[1]-circle3Pos[1]) - ja2 - ja1
    return np.array([ja1, ja2, ja3])

    # detect robot end-effector from the image
  def detect_end_effector(self,image):
    a = self.pixel2meter(image)
    endPos = a * (self.detect_yellow(image) - self.detect_red(image))
    return endPos

  # TODO: Define trajectory(Get answer from 3.1)
  def trajectory(self):
    # get current time
    cur_time = np.array([rospy.get_time() - self.time_trajectory])
    x_d = float(2.5* np.cos(cur_time * np.pi/15))
    y_d = float(2.5* np.sin(cur_time * np.pi/15))
    z_d = float(1* np.sin(cur_time * np.pi/15))
    return np.array([x_d, y_d, z_d])

  # detect robot end−effector from the image def detect end effector(self ,image):
  def detect_end_effector(self,image):
    a = self.pixel2meter(image)
    endPos = a * (self.detect_yellow(image) - self.detect_red(image)) 
    return endPos

  # Calculate the forward kinematics from image
  # joints_angles is a list of 4 angles for 1 end_effector position
  def forward_kinematics_image(self,joints_angles):
    joints = joints_angles
    q1 = joints[0]
    q2 = joints[1]
    q3 = joints[2]
    q4 = joints[3]

    k1 = 3*np.cos(q1)*(np.sin(q3)**2)+3*np.cos(q3)*np.sin(q1)*np.sin(q3)*np.cos(q2)+3*np.cos(q3)*np.sin(q1)*np.sin(q2)+3.5*np.cos(q1)*np.cos(q3)-3.5*np.sin(q1)*np.sin(q3)*np.cos(q2)+2.5*np.cos(q1)
    k2 = 3*np.sin(q1)*(np.sin(q3)**2)-3*np.cos(q1)*np.cos(q3)*np.sin(q3)*np.cos(q2)-3*np.cos(q1)*np.cos(q3)*np.sin(q2)+3.5*np.cos(q3)*np.sin(q1)+3.5*np.cos(q1)*np.sin(q2)*np.sin(q3)+2.5*np.sin(q1)
    k3 = -3*np.cos(q3)*np.sin(q3)*np.sin(q2)+3*np.cos(q3)*np.cos(q2)+3.5*np.sin(q3)*np.sin(q2)
    end_effector = np.array([k1,k2,k3])
    return end_effector

  # Calculate the Jacobian of the robot
  # joints_angles is a list of 4 angles for 1 end_effector position
  def calculate_jacobian(self,joints_angles):
    joints = joints_angles 
    q1 = joints[0]
    q2 = joints[1]
    q3 = joints[2]
    q4 = joints[3]

    J11 = -3*np.sin(q1)*(np.sin(q3)**2)+3*np.cos(q1)*np.cos(q3)*np.cos(q2)*np.sin(q3)+3*np.cos(q1)*np.cos(q3)*np.sin(q2)-3.5*np.sin(q1)*np.cos(q3)-3.5*np.cos(q1)*np.sin(q3)*np.cos(q2)-2.5*np.sin(q1)
    J12 = -3*np.cos(q3)*np.sin(q1)*np.sin(q3)*np.sin(q2)+3*np.cos(q3)*np.sin(q1)*np.cos(q2)+3.5*np.sin(q1)*np.sin(q3)*np.sin(q2)
    J13 = 6*np.cos(q1)*np.sin(q3)*np.cos(q3)-3*np.cos(q2)*np.sin(q1)*(np.cos(q3)**2)+3*np.cos(q2)*np.sin(q1)*(np.sin(q3)**2)+3*np.sin(q3)*np.cos(q1)+np.sin(q2)-3.5*np.sin(q3)*np.sin(q1)+3.5*np.cos(q3)*np.cos(q1)*np.sin(q2)
    J14 = 0

    J21 = 3*np.cos(q1)*(np.sin(q3)**2)+3*np.sin(q1)*np.cos(q3)*np.sin(q3)*np.cos(q2)+3*np.sin(q1)*np.cos(q3)*np.sin(q2)+3.5*np.cos(q3)*np.cos(q1)-3.5*np.sin(q1)*np.sin(q2)*np.sin(q3)+2.5*np.cos(q1)
    J22 = 3*np.cos(q1)*np.cos(q3)*np.sin(q3)*np.sin(q2)-3*np.cos(q1)*np.cos(q3)*np.cos(q2)+3.5*np.cos(q1)*np.cos(q2)*np.sin(q3)
    J23 = 6*np.sin(q1)*np.sin(q3)*np.cos(q3)+3*np.cos(q1)*np.cos(q2)*(np.sin(q3)**2)-3*np.cos(q1)*np.cos(q2)*(np.cos(q3)**2)+3*np.cos(q1)*np.sin(q3)*np.sin(q2)-3.5*np.sin(q3)*np.sin(q1)+3.5*np.cos(q1)*np.sin(q2)*np.cos(q3)
    J24 = 0

    J31 = 0
    J32 = -3*np.cos(q3)*np.sin(q3)*np.cos(q2)-3*np.cos(q3)*np.sin(q2)+3.5*np.sin(q3)*np.cos(q2)
    J33 = -3*np.sin(q2)*(np.cos(q3)**2)+3*np.sin(q2)*(np.sin(q3)**2)-3*np.sin(q3)*np.cos(q2)+3.5*np.cos(q3)*np.sin(q2)
    J34 = 0

    jacobian = np.array([[J11,J12,J13,J14],[J21,J22,J23,J24],[J31,J32,J33,J34]])
    return jacobian

  # Estimate control inputs for open-loop control
  def control_open(self,image):
    # estimate time step
    cur_time = rospy.get_time()
    dt = cur_time - self.time_previous_step2
    self.time_previous_step2 = cur_time
    q = self.detect_joint_angles(image) # estimate initial value of joints'
    J_inv = np.linalg.pinv(self.calculate_jacobian(image))  # calculating the psudeo inverse of Jacobian
    # desired trajectory
    pos_d= self.trajectory() 
    # estimate derivative of desired trajectory
    self.error_d = (pos_d - self.error)/dt
    self.error = pos_d
    q_d = q + (dt * np.dot(J_inv, self.error_d.transpose()))  # desired joint angles to follow the trajectory
    return q_d

  def control_closed(self,image):
    # P gain
    K_p = np.array([[10,0],[0,10]])
    # D gain
    K_d = np.array([[0.1,0],[0,0.1]])
    # estimate time step
    cur_time = np.array([rospy.get_time()])
    dt = cur_time - self.time_previous_step
    self.time_previous_step = cur_time
    # robot end-effector position
    pos = self.detect_end_effector(image)
    # desired trajectory
    pos_d= self.trajectory() 
    # estimate derivative of error
    self.error_d = ((pos_d - pos) - self.error)/dt
    # estimate error
    self.error = pos_d-pos
    q = self.detect_joint_angles(image) # estimate initial value of joints'
    J_inv = np.linalg.pinv(self.calculate_jacobian(image))  # calculating the psudeo inverse of Jacobian
    dq_d =np.dot(J_inv, ( np.dot(K_d,self.error_d.transpose()) + np.dot(K_p,self.error.transpose()) ) )  # control input (angular velocity of joints)
    q_d = q + (dt * dq_d)  # control input (angular position of joints)
    return q_d

  # Recieve data, process it, and publish
  def callback(self,data):
    # Recieve the image
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    # Perform image processing task (your code goes here)
    # The image is loaded as cv_imag

    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    cv2.imshow('window', cv_image)
    cv2.waitKey(3)
    
    # publish robot joints angles (lab 1 and 2)
    self.joints=Float64MultiArray()
    self.joints.data= self.detect_joint_angles(cv_image)

    
    # compare the estimated position of robot end-effector calculated from images with forward kinematics(lab 3)
    x_e = self.forward_kinematics(cv_image)
    x_e_image = self.detect_end_effector(cv_image)
    self.end_effector=Float64MultiArray()
    self.end_effector.data= x_e_image	

    # send control commands to joints (lab 3)
    q_d = self.control_closed(cv_image)
    #q_d = self.control_open(cv_image)
    self.joint1=Float64()
    self.joint1.data= q_d[0]
    self.joint2=Float64()
    self.joint2.data= q_d[1]
    self.joint3=Float64()
    self.joint3.data= q_d[2]

    # Publishing the desired trajectory on a topic named trajectory(lab 3)
    x_d = self.trajectory()    # getting the desired trajectory
    self.trajectory_desired= Float64MultiArray()
    self.trajectory_desired.data=x_d

    # Publish the results
    try: 
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
      self.joints_pub.publish(self.joints)
      self.end_effector_pub.publish(self.end_effector)
      self.trajectory_pub.publish(self.trajectory_desired)
      self.robot_joint1_pub.publish(self.joint1)
      self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
