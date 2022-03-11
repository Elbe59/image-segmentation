"""
======================
Apply maskSLIC vs SLIC
======================

This example is about comparing the segmentations obtained using the
plain SLIC method [1]_ and its masked version maskSLIC [2]_.

To illustrate these segmentation methods, we use an image of biological tissue
with immunohistochemical (IHC) staining. The same biomedical image is used in
the example on how to
:ref:`sphx_glr_auto_examples_color_exposure_plot_ihc_color_separation.py`.

The maskSLIC method is an extension of the SLIC method for the
generation of superpixels in a region of interest. maskSLIC is able to
overcome border problems that affects SLIC method, particularely in
case of irregular mask.

.. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
    Pascal Fua, and Sabine Süsstrunk, "SLIC Superpixels Compared to
    State-of-the-Art Superpixel Methods," IEEE TPAMI, 2012,
    :DOI:`10.1109/TPAMI.2012.120`

.. [2] Irving, Benjamin. "maskSLIC: regional superpixel generation
    with application to local pathology characterisation in medical
    images," 2016, :arXiv:`1606.09518`

"""

from os import listdir
import skimage.filters as filters
from skimage import data, io, segmentation, color
from skimage.future import graph
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure

from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def test():
    # Create the dimensions of the figure and set title and color:
    fig = plt.figure(figsize=(11, 10))
    plt.suptitle("Otsu's binarization algorithm applying a Gaussian filter", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    # Load the image and convert it to grayscale:
    image = cv2.imread('./Images/Echantillion1Mod2_301.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Otsu's binarization algorithm:
    ret1,th1 = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
    #ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #  Blurs the image using a Gaussian filter to eliminate noise
    gray_image_blurred = cv2.GaussianBlur(gray_image, (25, 25), 0)

    # Calculate histogram after filtering:
    hist2 = cv2.calcHist([gray_image_blurred], [0], None, [256], [0, 256])

    # Otsu's binarization algorithm:
    ret2, th2 = cv2.threshold(gray_image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Plot all the images:
    show_img_with_matplotlib(image, "image with noise", 1)
    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img with noise", 2)
    show_img_with_matplotlib(cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR),
                             "Otsu's binarization (before applying a Gaussian filter)", 4)
    show_img_with_matplotlib(cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR),
                             "Otsu's binarization (after applying a Gaussian filter)", 6)

    # Show the Figure:
    plt.show()


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])



def other_method():
    g = graph.rag_mean_color(img, slic)

    labels2 = graph.merge_hierarchical(slic, g, thresh=30, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)
    out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
    print(len(out))
    plt.imshow(out)
    plt.show()



def img_load():
    """
    Description :
    Méthode pour le chargement des images considérant le répertoire passé en paramètre.
    """

    img_list = {}

    for img in listdir('./Images/'):
        img_list[img] = cv2.imread('./Images/' + img)
        # img_list[img] = cv2.resize(img_list[img], DIM_IMG)

    # img_ref = cv2.resize(img_ref, DIM_IMG)

    return img_list



# Input data
img_list = img_load()

for img_name, img in img_list.items():
    #test()

    # plt.imshow(filtered)
    # plt.show()
    # mask
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(ret)
    # plt.imshow(thresh)
    # plt.show()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray_image = cv2.bilateralFilter(gray_image, 20, 60, 60)
    # gray_image = cv2.GaussianBlur(gray_image, (25, 25), 0)

    #HSV = cv2.cvtColor(image_blurred,cv2.COLOR_BGR2HSV)
    #gray_image = cv2.bilateralFilter(HSV, 5, 10,10)
    #gray_image = cv2.medianBlur(gray_image,9)

    # plt.imshow(gray_image)
    # plt.show()
    # continue
    gray_image = np.where(gray_image > 25, gray_image+10, 0) # Augmentation de la luminosité de l'image en nuance de gris

    # lum = color.rgb2gray(img)
    ## Création du masque en enlevant les petits morceaux et avec une certaine valeur minimum de gris, basé sur
    ## la valeur threshold d'un histogram.
    mask = morphology.remove_small_holes(
        morphology.remove_small_objects(
            gray_image > 30, 500),
        500)

    mask = morphology.opening(mask, morphology.disk(10))   ## Système d'érosion, dilation afin de marquer les différences entre lumineux et sombre
    #
    # SLIC result
    slic = segmentation.slic(img, n_segments=30, start_label=1)

    # maskSLIC result
    m_slic = segmentation.slic(img, n_segments=30, mask=mask, start_label=1)
    print("SLIC number of segments: %d" % len(np.unique(m_slic)))
    #m_slic = np.unique(m_slic)
    regions = measure.regionprops(m_slic, intensity_image=img)
    # print([r.intensity_mean for r in regions])

    data = pd.DataFrame(columns=["Grain isolé n°","Moyenne de B", "Moyenne de G", "Moyenne de R"])
    print(data)
    print(len(np.unique(m_slic)))
    for i in range(len(np.unique(m_slic))-1):
        r = regions[i]
        data.loc[i] = ['Grain isolé n°' + str(i+1)] + [r.intensity_mean[0]] + [r.intensity_mean[1]] + [r.intensity_mean[2]]


    print(data)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original image')
    axes[0, 0].set_axis_off()

    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Mask')
    axes[0, 1].set_axis_off()

    axes[1, 0].imshow(m_slic, cmap=plt.cm.tab20b)
    #axes[1, 0].imshow(segmentation.mark_boundaries(img, slic))
    #axes[1, 0].contour(mask, colors='red', linewidths=1)
    axes[1, 0].set_title('SLIC')
    axes[1, 0].set_axis_off()

    axes[1, 1].imshow(segmentation.mark_boundaries(img, m_slic))
    axes[1, 1].contour(mask, colors='red', linewidths=1)
    axes[1, 1].set_title('maskSLIC')
    axes[1, 1].set_axis_off()

    plt.tight_layout()
    plt.show()



