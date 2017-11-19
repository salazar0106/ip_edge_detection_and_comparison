import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('f.jpg',0)
edges = cv2.Canny(img,100,200)
img_gaussian = cv2.GaussianBlur(img,(3,3),0)

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)


cv2.imshow("Prewitt X", img_prewittx)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
cv2.imshow("Prewitt Y", img_prewitty)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
cv2.imshow("Prewitt", img_prewittx + img_prewitty)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()