import cv2 as cv
import numpy as np

# Local imports
import joints_locator

# Link lengths in meters
LINK_L = [2.5, 0, 3.5, 3]

# Initial images path (assumes running from catkin_ws)
IMG1_INIT = './src/ivr_assignment/images/image1_init.png'
IMG2_INIT = './src/ivr_assignment/images/image2_init.png'
ML_DATA_DIR = './src/ivr_assignment/images/ml_samples/'

# To be calculated during initialization
PIXELS_TO_METERS_COEFFICIENT_ = None

# Stationary joints locations
YELLOW_PIXEL_LOCATION = np.array([399, 555])
BLUE_PIXEL_LOCATION = np.array([399, 488])

def get_link_length(x):
    """Returns the length (in meters) of the specified link.
    For example, get_link_l(1) returns 2.5"""
    return LINK_L[x-1]

def get_pixels_to_meters_coefficient():
    """Safe function used to get the coefficient for converting image pixels to meters"""
    global PIXELS_TO_METERS_COEFFICIENT_
    if PIXELS_TO_METERS_COEFFICIENT_ is None:
        # Calculate the coefficient from the sample images if not initialized
        img1 = cv.imread(IMG1_INIT, cv.IMREAD_COLOR)
        img2 = cv.imread(IMG2_INIT, cv.IMREAD_COLOR)
        jl = joints_locator.joints_locator()
        locs = jl.get_joints_xyz_locations(img1, img2)
        # z-coordinate of red joint
        r_z = locs[3, 2]
        PIXELS_TO_METERS_COEFFICIENT_ = np.sum(LINK_L) / r_z
    return PIXELS_TO_METERS_COEFFICIENT_

def get_meters_to_pixels_coefficient():
    """Safe function used to get the coefficient for converting meters to image pixels"""
    return 1/get_pixels_to_meters_coefficient()
