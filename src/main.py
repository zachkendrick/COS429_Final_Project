'''------------------------------------------------------
Authors: Zachary Kendrick, Chloe Song, Katy Ho
Class: Princeton University COS429
Project: Augmented Reality

This file provides the primary functionality for the AR
project. It initializes a video capture sequence that takes
in a USB port as a parameter. Port 0 should be the port 
for most forward facing cameras on laptops. If you are using
a webcam connected to a USB port try ports 1,2,3,etc.

A frame should be displayed showing the output video feed.
Exit out of this program by pressing the 'q' key. 

---------------------------------------------------------'''

import numpy as np
import cv2

# USB port or file name
PORT_NUMBER = 0

# Video input from webcam USB port or file
cap = cv2.VideoCapture(PORT_NUMBER)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''-------------------------------------------------
	
	

	Calls to classes that manipulate frames go here!




    -------------------------------------------------'''

    # Display the resulting frame
    cv2.imshow('frame',gray_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()