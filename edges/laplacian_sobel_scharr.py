import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('f.jpg',0)

laplacian = cv2.Laplacian(img,cv2.CV_8U)
sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
scharrx = cv2.Scharr(img,cv2.CV_8U,1,0)
scharry = cv2.Scharr(img,cv2.CV_8U,0,1)


cv2.imshow('dst',laplacian)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
    
cv2.imshow('dst',sobelx)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

cv2.imshow('dst',sobely)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
    
cv2.imshow('dst',sobelx+sobely)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
    
cv2.imshow('dst',scharrx)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

cv2.imshow('dst',scharry)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()



cv2.imshow('dst',scharrx+scharry)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()