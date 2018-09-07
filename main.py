from subprocess import call
import pylab
import cv2
import numpy as np
import glob
#from surfnet import *
from surfnet_partial import *
#from surfnet_r2n2 import *
import math

def resize_images(path):
    images = glob.glob(path+'*.jpg')
    i = 1
    for fname in images:
        img = cv2.imread(fname)
        h, w, dim = img.shape
        if h!=128 and w!=128:
            img = cv2.resize(img, (128,128))

        if dim==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #command = "rm " + fname
        #call(command, shell=True)
        path2 = '/home/anneke/project/surf/tensorflow/data/train/'
        cv2.imwrite(path2+'frame'+str(i)+'.jpg', img);
        i = i + 1

def resize_video_frame(path, height, width):
    vidcap = cv2.VideoCapture(path+'/video/train.webm')
    vidcap.set(0,5000)
    frameRate = vidcap.get(5)

    if (vidcap.isOpened()== False):
        print("Error opening video stream or file")

    i = 1
    # Read until video is completed
    while(vidcap.isOpened()):
        frameId = vidcap.get(1)
        (success, img) = vidcap.read()
        if success == False :
            return
        if frameId % math.floor(frameRate)==0:
            h,w,dim = img.shape
            if h!=128 and w!=128:
                img =  cv2.resize(img, (height, width), interpolation = cv2.INTER_LINEAR)

            #if dim==3:
            #    img = cv2.cvtColor(img, CV_LOAD_IMAGE_COLOR)
            #print('Read a new frame: ', success)
            cv2.imwrite(path+"/train2/frame%d.jpg" % i, img)      # save frame as JPEG file
            i = i + 1

# Resize all Testing Images
path = '/home/anneke/project/surf/tensorflow/data/images/'
#resize_images(path=path)

# Resize video resolution
path_video = '/home/anneke/project/surf/tensorflow/data'
#resize_video_frame(path_video, 128, 128)

mode = 'train'
net = cnn4sft(mode)
