import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('f.jpg',0)
edges = cv2.Canny(img,100,200)


cv2.imshow('dst',edges)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()