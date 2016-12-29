import numpy as np
import cv2
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from webcam import Webcam
import signal
import sys

class Test:
 
    def _draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
        # draw teapot
        glutSolidTeapot(0.5)
 
        glutSwapBuffers()

    def main(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(0,0);
        self.win = glutCreateWindow("COS 429 AR Project")
        glutDisplayFunc(self._draw_scene)
        glutMainLoop()

# run instance of Hand Tracker 
test = Test()
test.main()