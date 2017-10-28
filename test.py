from skimage.draw import circle_perimeter
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import numpy as np
import pylab
import matplotlib.pyplot as plt

def test1():
    img = np.zeros((100, 100))
    rr, cc = circle_perimeter(35, 45, 25)
    img[rr, cc] = 1

    img = gaussian(img, 2)
    pylab.imshow(img)

    s = np.linspace(0, 2*np.pi,100)
    init = 50*np.array([np.cos(s), np.sin(s)]).T+50

    plt.plot(init[:,0],init[:,1])
    snake = active_contour(img, init, w_edge=0, w_line=1,max_iterations=2500)
    plt.plot(snake[:,0],snake[:,1])
    plt.show()

import cv2
from PIL import Image
def test2():
    img=Image.open('toyobjects.png','r')

    s = np.linspace(0, 2*np.pi,100)
    init = 80*np.array([np.cos(s), np.sin(s)]).T+100

    bc = ['periodic', 'free', 'fixed']
    for k in range(3):
        snake=active_contour(img,init,w_edge=0,w_line=1,bc=bc[k],max_iterations=2500)

        fig=plt.figure('fig_%d'%(k+1))
        plt.imshow(img,cmap="gray")
        plt.plot(init[:, 0], init[:, 1])
        plt.plot(snake[:, 0], snake[:, 1],'r')
        #height, width = img.size
        #plt.xlim(0,width)
        #plt.ylim(0,height)
        fig.show()
        #plt.hold(True)
    plt.pause(10)

from npytomat import npy_to_matlab
def test_npytomat(name):
    npy_to_matlab(name)
#test2()
filename='cropped_frame'
test_npytomat(filename)