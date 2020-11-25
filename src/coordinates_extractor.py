#!/usr/bin/env python3

import cv2
import numpy as np

# local imports
import math_utils as mu

# Class used to extract object coordinates from images using color thresholding
class coordinates_extractor:

    def __init__(self):
        pass

    # Returns a binary image of a colored region in an image using thresholding
    # color_thresholds has shape (2,3)
    # color_thresholds[0,i] = min value for color i
    # color_thresholds[1,i] = max value for color i
    def get_blob_threshold(self, img, color_thresholds):
        assert color_thresholds.shape == (2, 3)
        return cv2.inRange(img, color_thresholds[0, :], color_thresholds[1, :])

    # Given a binary image of a blob calculates its centre in pixel coordinates
    def get_blob_coordinates(self, blob):
        # Apply dilate to make the region larger
        #kernel = np.ones((5, 5), np.uint8)
        #blob = cv2.dilate(blob, kernel, iterations=3)
        # Use moments to calculate centre of the blob
        centre = self.centre_of_mass_from_blob(blob)

        return centre

    def centre_of_mass_from_blob(self, blob):
        # Approximate c.o.m. using moments
        M = cv2.moments(blob)
        try:
            approx_cx = int(M['m10'] / M['m00'])
            approx_cy = int(M['m01'] / M['m00'])
        except Exception as _:
            return None
        c_o_m = np.array([approx_cx, approx_cy])
        return c_o_m



# test the class
def main():
    ce = coordinates_extractor()
    red_thresholds = np.array([[0, 0, 100], [10, 10, 255]])
    img = cv2.imread('image_copy.png', cv2.IMREAD_COLOR)
    blob = ce.get_blob_threshold(img, red_thresholds)
    print(ce.get_blob_coordinates(blob))

if __name__ == '__main__':
    main()
