import cv2
from cvzone.ColorModule import ColorFinder
import cvzone 
import time 
import math 
import numpy as np 
import sys

try:
    vid_num = sys.argv[1]
except:
    vid_num = 4
cap = cv2.VideoCapture(f'videos/vid ({vid_num}).mp4')


cFinder = ColorFinder(False)
hsv_value = {'hmin': 4, 'smin': 144, 'vmin': 63, 'hmax': 20, 'smax': 255, 'vmax': 192}

ball_pos_x, ball_pos_y = [], []
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
PURPLE = (255, 0, 255)
RED = (0, 0, 255)

frame_width = 1300
xvals = [item for item in range(0, frame_width)]

# Quadratic coefficients
A = B = C = -1
prediction = 0

while True:
    # video feed
    ret, frame = cap.read()

    # image feed
    # frame = cv2.imread('Ball.png')
    frame = frame[:900, :]

    # Filter color of ball from environment
    imgColor, mask = cFinder.update(frame, hsv_value)

    # Track ball based on color contour
    imgContours, contours = cvzone.findContours(frame,mask, minArea=200)

    '''
    Since the trajectory of the ball is a ballistic one
    Polynomial regression is used to model the behaviour 
    Thus the quadratic formula or Almighty formula
    '''

    for cnt in contours:
        ball_pos_x.append(contours[0]['center'][0])
        ball_pos_y.append(contours[0]['center'][1])
        A, B, C = np.polyfit(ball_pos_x, ball_pos_y, 2)


    # drawing predictions lines on video feed
    for i, (x, y) in enumerate(zip(ball_pos_x, ball_pos_y)):
        pos = (x,y)
        cv2.circle(frame, pos, 8, GREEN, cv2.FILLED)
        if i == 0:
            cv2.line(frame, pos, pos, BLUE, 5)
        else:
            cv2.line(frame, pos, (ball_pos_x[i-1], ball_pos_y[i-1]), BLUE, 5)
    
    # using our polynomial regression to predict trajectory
    for x in xvals:
        y = int(A*x**2 + B*x + C)
        cv2.circle(frame,(x,y),4, PURPLE)

    '''
    predicting our landing point on the x coordinate
    using the almighty formula
    '''
    a = A
    b = B
    c = C - 590

    if len(ball_pos_x) > 10:
        dis = b * b - 4 * a * c
        sqrt_val = math.sqrt(abs(dis))
        x = int((-b - sqrt_val) / (2*a))
        print(x)
        prediction = 330 < x < 430

        if prediction:
            cvzone.putTextRect(frame, 'Score', (100, 100), colorR=GREEN,offset=12)
        else:
            cvzone.putTextRect(frame, 'Miss', (100, 100), colorR=RED,offset=12)

    # show ball with tracking 
    cv2.imshow('frame', frame)
    
    cv2.waitKey(40) 

