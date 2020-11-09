#!/usr/bin/env python3

import sys
import cv2
import numpy as np

# Class used to extract object coordinates from images using color thresholding
class coordinates_extractor:

  def __init__(self):
    pass

  # Returns the pixel coordinates of the center of a colored region in an image
  # color_thresholds has shape (2,3)
  # color_thresholds[0,i] = min value for color i
  # color_thresholds[1,i] = max value for color i
  def get_coordinates(self, img, color_thresholds):
    assert color_thresholds.shape == (2,3)
    # Isolate the desired color as a binary image
    mask = cv2.inRange(img, color_thresholds[0,:], color_thresholds[1,:])
    # Apply dilate to make the region larger
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    # Use moments to calculate centre of the blob
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return np.array([cx, cy])

# test the class
def main(args):
  ce = coordinates_extractor()
  red_thresholds = np.array([[0,0,100], [10,10,255]])
  img = cv2.imread('image_copy.png', cv2.IMREAD_COLOR)
  print(ce.get_coordinates(img, red_thresholds))

if __name__ == '__main__':
    main(sys.argv)
