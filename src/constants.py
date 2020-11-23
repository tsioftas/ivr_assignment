import cv2 as cv
import numpy as np


# Local imports
import joints_locator

# Link lengths in meters
LINK_L = [2.5, 0, 3.5, 3]

# Initial images path
IMG1_INIT = './src/ivr_assignment/images/image1_init.png'
IMG2_INIT = './src/ivr_assignment/images/image2_init.png'

# To be calculated during initialization
PIXELS_TO_METERS_COEFFICIENT = None

def get_link_l(x):
    return LINK_L[x-1]

def get_pixels_to_meters_coefficient():
    if PIXELS_TO_METERS_COEFFICIENT is None:
        img1 = cv.imread(IMG1_INIT, cv.IMREAD_COLOR)
        img2 = cv.imread(IMG2_INIT, cv.IMREAD_COLOR)
        jl = joints_locator.joints_locator()
        locs = jl.get_joints_xyz_locations(img1, img2)
        r_z = locs[3,2]
        PIXELS_TO_METERS_COEFFICIENT = np.sum(LINK_L) / r_z
    return PIXELS_TO_METERS_COEFFICIENT
