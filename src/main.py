"""
Authors: Zachary Kendrick, Chloe Song, Katy Ho
Class: Princeton University COS429
Project: Augmented Reality

This file provides the primary functionality for the AR
project. It initializes a video capture sequence that takes
in a USB port as a parameter. Port 0 should be the port 
for most forward facing cameras on laptops. If you are using
a webcam connected to a USB port try ports 1,2,3,etc.

Run the code by simply running:

    $ python main.py

Then select from the segmentation algorith options.

A frame should be displayed showing the output video feed.
Exit out of this program by pressing the 'q' key. 

"""

import numpy as np
import cv2
from Preprocess import Preprocess as pre
from Hand import Hand as hand
import sys


# USB port or file name
PORT_NUMBER = 0

# Video input from webcam USB port or file
cap = cv2.VideoCapture(PORT_NUMBER)

# threshold boundaries for HSV space skin segmentation
lower = np.array([0, 0, 0], dtype = "uint8")
upper = np.array([255, 255, 255], dtype = "uint8")

# video recording 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('COS429.avi',fourcc, 20.0, (640,480))

# Choose the segmentation algorithm
option = int(raw_input("What skin segmentation algorithm would you like to use?\n1) OTSU Thresholding \n2) HSV Skin Sample Segmentation\n"))

while(True):
    # Capture frame-by-frameqq
    ret, BGR_frame = cap.read()

    # thresholding to find hand
    if option == 1:
        BGR_frame = pre.deNoise(BGR_frame)
        gray_frame = pre.im2Gray(BGR_frame)
        skinMask = pre.thresholdHand(gray_frame)

    # skin segmentation to find hand
    if option == 2:
        roi = pre.getHandROI(BGR_frame)
        if roi != None:
            lower, upper = pre.updateHSV(roi)
        skinMask = pre.getSkinMask(lower, upper, BGR_frame)
        BGR_frame = pre.applyMask(skinMask, BGR_frame)

    # find finger tips and palm
    cnt, hull = hand.getConvexHull(skinMask)
    cv2.drawContours(BGR_frame,[hull],0,(255,0,0),2)
    palmCenter = hand.centerOfPalm(cnt)
    cv2.circle(BGR_frame, palmCenter, 5, [0,0,255], 2)

    # Display the resulting frame
    cv2.imshow('frame', BGR_frame)

    # Record a video
    # out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()