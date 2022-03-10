from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.io import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage import exposure, morphology
from skimage.segmentation import watershed, mark_boundaries, quickshift

img = imread('./Images/Echantillion1Mod2_301.png')

# Contrast stretching
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

# Adaptive Equalization
img_eq = exposure.equalize_adapthist(img, clip_limit=0.03)

# mean shift
shifted = cv2.pyrMeanShiftFiltering(img, 21, 31)

"""
cv2.imshow("Img", img)
cv2.imshow("Shifted", shifted)
cv2.waitKey(0)
cv2.destroyWindow()
"""

# threshold
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)[1]
thresh = morphology.opening(thresh, morphology.disk(5)) # pour watershed pour avoir des beaux cercles et moins de petits blops :)

"""
cv2.imshow("Img", img)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
cv2.destroyWindow()
"""

# Watershed
""""
# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
dist = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(dist, indices=False, min_distance=20, labels=thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-dist, markers, mask=thresh)
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
"""

# quickshift
"""
segments_quick = quickshift(shifted, kernel_size=10, max_dist=25, ratio=0.5)

print("Quickshift number of segments: %d" % len(np.unique(segments_quick)))

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, subplot_kw={'adjustable': 'box'})
fig.set_size_inches(8, 3, forward=True)
fig.tight_layout()

ax[0].imshow(img)
ax[0].set_title("Image")
ax[1].imshow(mark_boundaries(img, segments_quick))
ax[1].set_title("Quickshift")

for a in ax:
    a.set_xticks(())
    a.set_yticks(())
plt.show()
"""




