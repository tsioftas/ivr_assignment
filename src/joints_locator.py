#!/usr/bin/env python3

import sys
import cv2
import numpy as np

#local imports
from coordinates_extractor import coordinates_extractor

# Uses coordinates_extractor to get positions of joints in an image
class joints_locator:

  def __init__(self):
    self.blue_thresholds = np.array([[100,0,0], [255,10,10]])
    self.green_thresholds = np.array([[0,100,0], [10,255,10]])
    self.red_thresholds = np.array([[0,0,100], [10,10,255]])
    self.yellow_thresholds = np.array([[0,100,100], [10,255,255]])
    self.ce = coordinates_extractor()
    pass

  def get_blue_joint(self, img):
    blob = self.ce.get_blob_threshold(img, self.blue_thresholds)
    return self.ce.get_blob_coordinates(blob)

  def get_green_joint(self, img):
    blob = self.ce.get_blob_threshold(img, self.green_thresholds)
    return self.ce.get_blob_coordinates(blob)

  def get_red_joint(self, img):
    blob = self.ce.get_blob_threshold(img, self.red_thresholds)
    return self.ce.get_blob_coordinates(blob)
  
  def get_yellow_joint(self, img):
    blob = self.ce.get_blob_threshold(img, self.yellow_thresholds)
    # separate yellow from orange
    bin2 = img.copy()
    bin2[img[:,:,1] == img[:,:,2]] = 255
    bin2[img[:,:,1] != img[:,:,2]] = 0
    blob = blob & bin2[:,:,0]
    return self.ce.get_blob_coordinates(blob)

  def get_joints_location(self, img):
    y = self.get_yellow_joint(img)
    b = self.get_blue_joint(img)
    g = self.get_green_joint(img)
    r = self.get_red_joint(img)
    return np.array([y,b,g,r])

# test the class
def main(args):
  jl = joints_locator()
  img = cv2.imread('image_copy.png', cv2.IMREAD_COLOR)
  print(jl.get_joints_location(img))

if __name__ == '__main__':
    main(sys.argv)
