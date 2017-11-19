import numpy as np
from scipy.ndimage import distance_transform_edt

img = cv2.imread('f.jpg',0)

canny = cv2.Canny(img,100,200)
sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
scharrx = cv2.Scharr(img,cv2.CV_8U,1,0)
scharry = cv2.Scharr(img,cv2.CV_8U,0,1)
laplacian = cv2.Laplacian(img,cv2.CV_8U)
scharr=scharrx+scharry
sobel=sobelx+sobely
img_gaussian = cv2.GaussianBlur(img,(3,3),0)
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
prewitty = cv2.filter2D(img_gaussian, -1, kernely)
prewitt=prewittx+prewitty
robert= cv2.imread('eo.jpg',0)

DEFAULT_ALPHA = 1.0 / 9

def fom(img, img_gold_std, alpha = DEFAULT_ALPHA):
    """
    Computes Pratt's Figure of Merit for the given image img, using a gold
    standard image as source of the ideal edge pixels.
    """

    # To avoid oversmoothing, we apply canny edge detection with very low
    # standard deviation of the Gaussian kernel (sigma = 0.1).
    edges_img = cv2.Canny(img,20,50)
    edges_gold = cv2.Canny(img_gold_std,20,50)
    
    # Compute the distance transform for the gold standard image.
    dist = distance_transform_edt(np.invert(edges_gold))

    fom = 1.0 / np.maximum(
        np.count_nonzero(edges_img),
        np.count_nonzero(edges_gold))

    N, M = img.shape

    for i in xrange(0, N):
        for j in xrange(0, M):
            if edges_img[i, j]:
                fom += 1.0 / ( 1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(
        np.count_nonzero(edges_img),
        np.count_nonzero(edges_gold))    

    return fom

print fom(sobel, robert)