import numpy as np
import rospy

# detect robot end−effector from the image def detect end effector(self ,image):
def detect_end_effector(self,image):
    a = self.pixel2meter(image)
    endPos = a * (self.detect_yellow(image) - self.detect_red(image)) 
    return endPos


# Calculate the forward kinematics
def forward_kinematics(self,image):
    joints = self.detect_joint_angles(image)
    end_effector = np.array([2.5*np.sin(joints[0]) + 3.5*np.sin(joints[0] + joints[1]) + 3*np.sin(joints.sum()), 2.5*np.cos(joints[0]) + 3.5*np.cos(joints[0]+joints[1]) + 3*np.cos(joints.sum())])
    return end_effector

# Calculate the Jacobian of the robot
def calculate_jacobian(self,image):
    joints = self.detect_joint_angles(image) 
    jacobian = np.array([  
                          [2.5*np.cos(joints[0]) + 3.5*np.cos(joints[0]+joints[1]) + 3*np.cos(joints.sum()),
                           3.5*np.cos(joints[0]+joints[1]) + 3*np.cos(joints.sum()),
                           3*np.cos(joints.sum())
                          ],
                          [- 2.5*np.sin(joints[0]) - 3.5*np.sin(joints[0]+joints[1]) - 3*np.sin(joints.sum()),
                           - 3.5*np.sin(joints[0]+joints[1]) - 3*np.sin(joints.sum()), -3*np.sin(joints.sum())
                          ]
                        ]
                       )
    return jacobian

# Define trajectory
def trajectory(self):
    # get current time
    cur_time = np.array([rospy.get_time() - self.time_trajectory])
    x_d = float(6 * np.cos(cur_time*np.pi/100))
    y_d = float(6 + np.absolute(1.5*np.sin(cur_time * np.pi/100)))
    return np.array([x_d, y_d])

# Controller:
# Estimate control inputs for open−loop control 
def control_open (self,image):
    # estimate time 
    cur_time = rospy.get_time()
    dt = cur_time - self.time_previous_step2
    self.time_previous_step2 = cur_time
    # estimate initial value of joints ’
    q = self.detect_joint_angles(image)
    # calculating the psudeo inverse of Jacobian
    J_inv = np.linalg.pinv(self.calculate_jacobian(image))
    # desired trajectory
    pos_d= self.trajectory()
    # estimate derivative of desired trajectory
    self.error_d = (pos_d - self.error)/dt 
    self.error = pos_d
    # integrat to find desired joint angles to follow the trajectory
    q_d = q + (dt*np.dot(J_inv, self.error_d.transpose())) 
    return q_d


# initialize errors
self.time_previous_step2 = np.array([rospy.get_time()],dtype='float64')
# initialize error and derivative of error for trajectory tracking
self.error = np.array([0.0,0.0], dtype='float64') 
self.error_d = np.array([0.0,0.0], dtype='float64')


# send control commands to joints (lab 3)
q_d = self.control_open(cv_image) 
self.joint1 = Float64() 
self.joint1.data = q_d[0]
self.joint2 = Float64() 
self.joint2.data = q_d[1]
self.joint3 = Float64() 
self.joint3.data = q_d[2]


# Perform control:
def control_closed(self,image): 
    # P gain
    K_p = np.array([[10,0],[0,10]])
    # D gain
    K_d = np.array([[0.1,0],[0,0.1]])
    # estimate time step
    cur_time = np.array([rospy.get_time_()]) 
    dt = cur_time - self.time_previous_step
    self.time_previous_step = cur_time
    # robot end−effector position
    pos = self.detect_end_effector(image)
    # desired trajectory
    pos_d= self.trajectory ()
    # estimate derivative of error
    self.error_d = ((pos_d - pos) - self.error)/dt # estimate error
    self.error_= pos_d - pos
    q = self.detect_joint_angles(image) # estimate initial value of joints’
    # calculating the psudeo inverse of Jacobian
    J_inv = np.linalg.pinv(self.calculate_jacobian(image))
    # calculate angular velocity of joints
    dq_d =np.dot(J_inv,(np.dot(Kd,self.error_d.transpose())
    + np.dot(Kp,self.error.transpose()) ) )
    # take integral to find control inputs (angular position of joints)
    q_d = q + (dt * dq_d) 
    return q_d

# move joints
q_d = self.control_closed(cv_image) 
self.joint1=Float64() 
self.joint1.data= q_d[0]
self.joint2=Float64() 
self.joint2.data= q_d[1]
self.joint3=Float64() 
self.joint3.data= q_d[2]
