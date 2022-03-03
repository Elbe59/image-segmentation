import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import watershed, expand_labels
from skimage.color import label2rgb
from skimage.io import imread
from skimage.color import rgb2gray

from skimage.segmentation import felzenszwalb, slic, quickshift

img = imread('./Images/Echantillion1Mod2_471.png')
img_grey = rgb2gray(img)
"""print(img.shape)
print(img_grey.shape)"""

# Make segmentation using edge-detection and watershed.
edges = sobel(img_grey)

# Identify some background and foreground pixels from the intensity values.
# These pixels are used as seeds for watershed.
markers = np.zeros_like(img_grey)
foreground, background = 1, 2
markers[img_grey < 40.0/255] = background
markers[img_grey > 40.0/255] = foreground

ws = watershed(edges, markers, connectivity=10)
seg1 = label(ws == foreground)

# Show the segmentations.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5), sharex=True, sharey=True)

axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[1].imshow(edges)
axes[1].set_title('Sobel')
color2 = label2rgb(seg1, image=img_grey, bg_label=0)
axes[2].imshow(color2)
axes[2].set_title('Sobel + Watershed')


for a in axes:
    a.axis('off')
fig.tight_layout()
plt.show()
