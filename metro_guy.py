import cv2
import numpy as np
from matplotlib import pyplot as plt

template = cv2.imread('metro_guy.jpg',0)
w, h = template.shape[::-1]

cap = cv2.VideoCapture('metro.mp4')

while(1):
    ret, frame = cap.read()
    if ret:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.75
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
        cv2.imshow('metro', frame)
        
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
        
cap.release()
cv2.destroyAllWindows()
