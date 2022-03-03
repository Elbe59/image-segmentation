# import the necessary packages
import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.io import imread
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


# load the image and perform pyramid mean shift filtering
# to aid the thresholding step
img = imread('./Images/Echantillion1Mod2_471.png')
shifted = cv2.pyrMeanShiftFiltering(img, 21, 31)
"""
cv2.imshow("filter", shifted)
cv2.waitKey(0)
cv2.destroyWindow()
"""

# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
"""
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
"""
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
"""
cv2.imshow("Img", img)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
cv2.destroyWindow()
"""

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)


# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

# display
fig, axes = plt.subplots(ncols=2, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Image')
ax[1].imshow(labels, cmap=plt.cm.tab20b)
ax[1].set_title('Segmentation')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
