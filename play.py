import cv2
import numpy as np
import os
import time

# get image from saved path
def get_image(path):
    image = cv2.imread(path)
    return image

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 800, 600)

