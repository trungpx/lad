import cv2
import numpy as np
import ipdb, time
t0 = time.time()
cap = cv2.VideoCapture('1.avi')
count = 0
while (True):
    ret, img = cap.read()
    #ipdb.set_trace()
    res = cv2.resize(img, (320,240))
    #ipdb.set_trace()
    cimg = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT, 1, 22, param1=330, param2=20, minRadius=15, maxRadius=25)
    try:
        l = len(circles[0,:])
        for i in circles[0,:]:
            cv2.circle(res,(i[0],i[1]),20,(0,0,255),5)
            cv2.circle(res,(i[0],i[1]),2, (10,255,10),5)
    except TypeError:  pass
    cv2.imshow('detected circles',res)
    cv2.waitKey(5)
    count+=1
    print(count); print("Elapsed: %s"%(time.time()-t0))
cap.release()
cv2.destroyAllWindows()
