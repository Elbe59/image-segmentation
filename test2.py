import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel

# Generate an initial image
img = imread('./Images/Echantillion1Mod2_471.png')
img_grey = rgb2gray(img)
print(img_grey)

edges = sobel(img_grey*255)

# Identify some background and foreground pixels from the intensity values.
# These pixels are used as seeds for watershed.
markers = np.zeros_like(img_grey, dtype=int)
foreground, background = 1, 0
markers[edges < 5] = background
markers[edges > 5] = foreground
plt.imshow(markers)

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(markers)
coords = peak_local_max(distance, footprint=np.ones((11, 11)), labels=markers)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(distance, markers, mask=img_grey)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()