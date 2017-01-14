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

from Axis import Axis
import time

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

ax = Axis()
curr_time = time.time()
ct = 1

print "Press c to use current frame for calibration"
while(True):
    # Capture frame-by-frameqq
    ret, BGR_frame = cap.read()
    BGR_frame = cv2.flip(BGR_frame,1) #vertical flip

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
    cnt, hull,indices = hand.getConvexHull(skinMask)
    cv2.drawContours(BGR_frame,[hull],0,(255,0,0),2)
    palmCenter = hand.centerOfPalm(cnt)
    cv2.circle(BGR_frame, palmCenter, 5, [0,0,255], 2)

    # Display the resulting frame
    #cv2.imshow('frame', BGR_frame)

    # Record a video
    # out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):

        cv2.imwrite("calibration.jpg", BGR_frame)
        print "Calibrating"
        fingers = ax.calibrate(BGR_frame, hull, indices, cnt, palmCenter)

        for finger in fingers:
            cv2.circle(BGR_frame, finger, 5, [0,0,255], 2)
        
        img_pts = ax.get_img_pts(fingers, palmCenter)
        axis = ax.get_axis_2d(img_pts)
        cv2.line(BGR_frame,axis[3],axis[0],(255,0,0),5)
        cv2.line(BGR_frame,axis[3],axis[1],(0,255,0),5)
        cv2.line(BGR_frame,axis[3],axis[2],(0,0,255),5)
        cv2.imshow('frame', BGR_frame)
        curr_time = time.time()

    else:

        if ax.calibrated:
            # calibrating again each time actually seems to work better idk
            #fingers = ax.calibrate(BGR_frame, hull, palmCenter)
            fingers = ax.find_fingers(BGR_frame, hull, indices, cnt, palmCenter)#ax.calibrate(BGR_frame, hull, palmCenter)

            for finger in fingers:
                cv2.circle(BGR_frame, finger, 5, [0,0,255], 2)
            
            if len(fingers) < 5:
                axis = ax.axis_2d
            else:
                img_pts = ax.get_img_pts(fingers, palmCenter)
                axis = ax.get_axis_2d(img_pts)

            cv2.fillPoly(BGR_frame,[np.array([axis[1],axis[3],axis[0],axis[6]])],(225,202,0))
            cv2.fillPoly(BGR_frame,[np.array([axis[1],axis[3],axis[2],axis[4]])],(225,202,0))
            cv2.fillPoly(BGR_frame,[np.array([axis[1],axis[4],axis[7],axis[6]])],(225,202,0))
            cv2.fillPoly(BGR_frame,[np.array([axis[2],axis[3],axis[0],axis[5]])],(225,202,0))
            cv2.fillPoly(BGR_frame,[np.array([axis[2],axis[4],axis[7],axis[5]])],(225,202,0))
            cv2.fillPoly(BGR_frame,[np.array([axis[7],axis[5],axis[0],axis[6]])],(225,202,0))

            cv2.line(BGR_frame,axis[3],axis[0],(0,0,0),5)
            cv2.line(BGR_frame,axis[3],axis[1],(0,0,0),5)
            cv2.line(BGR_frame,axis[1],axis[6],(0,0,0),5)

            cv2.line(BGR_frame,axis[3],axis[2],(0,0,0),5)

            cv2.line(BGR_frame,axis[1],axis[4],(0,0,0),5)

            cv2.line(BGR_frame,axis[4],axis[2],(0,0,0),5)
            cv2.line(BGR_frame,axis[4],axis[7],(0,0,0),5)

            cv2.line(BGR_frame,axis[7],axis[6],(0,0,0),5)
            cv2.line(BGR_frame,axis[7],axis[5],(0,0,0),5)
            cv2.line(BGR_frame,axis[5],axis[2],(0,0,0),5)
            cv2.line(BGR_frame,axis[5],axis[0],(0,0,0),5)

            cv2.line(BGR_frame,axis[0],axis[6],(0,0,0),5)

            #cv2.rectangle(BGR_frame, axis[3], axis[6],(0,255,0))
            #BGR_frame = cv2.drawContours(BGR_frame, [axis[1],axis[3],axis[6],axis[0]],0,(0,255,0),2)

        #     self.axis[3] = (origin[0], origin[1], 0) #(0,0,0)
        # self.axis[0] = (py[0], py[1], 0) #(0,3,0)
        # self.axis[1] = (px[0], px[1], 0) #(3,0,0)
        # self.axis[2] = (origin[0], origin[1], -200) #(0,0,-3)

        # self.axis[4] = (px[0], px[1], -200) #(3,0,-3)
        # self.axis[5] = (py[0], py[1], -200)  #(0,-3,-3)
        # self.axis[6] = (px[0], py[1], 0) #(3,3,0)
        # self.axis[7] = (px[0], py[1], -200) #(3,3,-3)

            # if time.time() - curr_time >= 2:
            #     cv2.imwrite("img" + str(ct) + ".jpg", BGR_frame)
            #     ct += 1
            #     print "saved image"
            #     curr_time = time.time()

        cv2.imshow('frame', gray_frame)
        #cv2.imshow('frame', BGR_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
#option = int(raw_input("What skin segmentation algorithm would you like to use?\n1) OTSU Thresholding \n2) HSV Skin Sample Segmentation\n"))

cap.release()
cv2.destroyAllWindows()