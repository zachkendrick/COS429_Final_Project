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
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from webcam import Webcam
import signal
import sys

class AR_Project:
 
    def __init__(self):
        # sigint interrupt initialize
        signal.signal(signal.SIGINT, self.signal_handler)    

        # initialize webcam
        self.webcam = Webcam()
        self.webcam.start()
          
        self.x_axis = 0.0
        self.y_axis = 0.0
        self.z_axis = 0.0
        self.z_pos = -7.0
        
        self.win = 0
        self.texture_background = None
        self.texture_teapot = None
     
    def signal_handler(self, signal, frame):
        print('\nYou pressed Ctrl+C!')
        self.webcam.close()
        sys.exit()

    def _get_background(self):
        # get image from webcam 
        image = self.webcam.get_current_frame()

        # convert image to OpenGL texture format
        image = cv2.flip(image, 0)
        image = cv2.flip(image, 1)
        gl_image = Image.fromarray(image)     
        ix = gl_image.size[0]
        iy = gl_image.size[1]
        gl_image = gl_image.tobytes("raw", "BGRX", 0, -1)
      
        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, gl_image)
 
    def _draw_background(self):
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 4.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 4.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 4.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 4.0)
        glEnd()
 
    def _init_gl(self, Width, Height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(Width)/float(Height)-.2, 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
 
        # enable texture
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)
 
        # initialize lighting 
        #glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))
        #glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.8, 0.0, 1.0)) 
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
 
        # initialize blending
        glColor4f(0.2, 0.2, 0.2, 0.5)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_BLEND)
 
        #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        #glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        #glEnable(GL_TEXTURE_2D)
 
    def _draw_scene(self):
        # handle any hand gesture
        self._get_background()
 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
            
        # draw background
        #glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()
        glTranslatef(0.0,0.0,-11.2)
        self._draw_background()
        glPopMatrix()

        # position teapot
        glTranslatef(0.0,0.0,self.z_pos);
        glRotatef(self.x_axis,1.0,0.0,0.0)
        glRotatef(self.y_axis,0.0,1.0,0.0)
        glRotatef(self.z_axis,0.0,0.0,1.0)
 
        # draw teapot
        glutSolidTeapot(1.2)
 
        # rotate teapot 
        self.x_axis = self.x_axis - 2
        self.z_axis = self.z_axis - 2
 
        glutSwapBuffers()
 
    def main(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(800, 400)
        self.win = glutCreateWindow("COS 429 AR Project")
        glutDisplayFunc(self._draw_scene)
        glutIdleFunc(self._draw_scene)
        self._init_gl(640, 480)
        glutMainLoop()
 
# run instance of Hand Tracker 
test = AR_Project()
test.main()