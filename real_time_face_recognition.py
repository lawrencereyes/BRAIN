#!/usr/bin/env python

'''
Face detection using haar cascades.

It is good for face recognition at real-time because it uses a camara to record a video and then takes a picture of the user.

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

#TODO: Integrate face recognition to this module. Use the opencv 

# Python 2/3 compatibility
from __future__ import print_function

# It allows me to import cv2
import sys
sys.path.append('/usr/local/lib/python3.5/site-packages')

import numpy as np
import cv2

# local modules
#TODO: Understand how they work
from video import create_capture
from common import clock, draw_str


def detect(img, cascade):
    '''Detects the shape given the classifier (haarcascade).'''
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, 
                                        minNeighbors=4, 
                                        minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    '''Draws the shape around the part or thing we are looking for in the picture.'''
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    # Gets the haarcascade
    cascade_fn = args.get('--cascade', "../OpenCV/data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../OpenCV/data/haarcascades/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('face_detection', vis)

        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()