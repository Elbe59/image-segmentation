from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.io import imread
import cv2


img = imread('./Images/Echantillion1Mod2_301.png')
filtered = cv2.bilateralFilter(img, 20, 60, 60)
filtered = cv2.pyrMeanShiftFiltering(filtered, 21, 31)


segments_quick = quickshift(filtered, kernel_size=11, max_dist=20, ratio=0.5)

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





