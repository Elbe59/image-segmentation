import sys
import warnings
from os import listdir
import numpy as np
import cv2
from scipy import ndimage
from skimage import exposure, morphology, segmentation, measure
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import pandas as pd
from skimage.color import label2rgb



def load_img():
    """
    Description :
    Méthode pour le chargement des images.
    """

    img_list = {}

    for img in listdir('./Images/'):
        img_list[img] = cv2.imread('./Images/' + img)

    return img_list


def save_df(df, img_name):
    """
    Description :
    Méthode pour la sauvegarde des dataframes au format csv.
    """

    df.to_csv("./Output/Dataframes/" + img_name.split(".")[0] + '.csv', index=False, header=True)


def save_results(img, img_name):
    """
    Description :
    Méthode pour la sauvegarde des images segmentées.
    """

    plt.imsave("./Output/Images/" + img_name.split('.')[0] + "_segmented." + img_name.split('.')[1], img[:, :, ::-1])


def display_segments(labels, img):
    """
    Description :
    Méthode pour l'affichage des segments.
    """

    for i, segVal in enumerate(np.unique(labels)):
        
        print("[x] inspecting segment %d" % i)

        mask = np.zeros(img.shape[:2], dtype="uint8")
        mask[labels == segVal] = 255

        cv2.imshow("Applied", cv2.bitwise_and(img, img, mask=mask))
        cv2.waitKey(0)


def sort_segments(labels):
    """
    Description :
    Méthode pour trier les segments, ceux don't l'aire ne dépasse pas 100 sont supprimés.
    """

    lb_idx = 1
    for region in regionprops(labels):
        # print(region.area)
        if region.area < 100:
            labels[labels == lb_idx] = 0
        lb_idx += 1


def calc_mean(labels, img):
    """
    Description :
    Méthode pour le calcul des moyennes sur les channels BGR de l'image de base en fonction des segments identifiés.
    """

    regions = measure.regionprops(labels, intensity_image=img)
    data = pd.DataFrame(columns=["Grain isolé n°", "Moyenne de B", "Moyenne de G", "Moyenne de R"])
    for i in range(len(np.unique(labels)) - 1):
        r = regions[i]
        data.loc[i] = ['Grain isolé n°' + str(i + 1)] \
                      + [round(r.intensity_mean[0], 2)] \
                      + [round(r.intensity_mean[1], 2)] \
                      + [round(r.intensity_mean[2], 2)]

    return data


def main():

    # args
    display = True

    # warnings suppression
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # load image
    img_list = load_img()

    for img_name, img in img_list.items():

        #--- contrast stretching ---
        p2, p98 = np.percentile(img, (2, 98))
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

        #--- mean shift ---
        shifted = cv2.pyrMeanShiftFiltering(img_rescale, 21, 31)

        #--- threshold ---
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]

        #--- erode/dilate ---
        thresh = morphology.opening(thresh, morphology.disk(5))

        #--- Watershed ---
        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        dist = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(dist, indices=False, min_distance=20, labels=thresh)

        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-dist, markers, mask=thresh)

        #--- segments sorting ---
        sort_segments(labels)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

        #--- rgb mean ---
        data = calc_mean(labels, img)

        #--- save results ---
        save_df(data, img_name)
        save_results(segmentation.mark_boundaries(img, labels), img_name)

        #--- display ---
        if display:
            fig, axes = plt.subplots(ncols=4, figsize=(12, 3))
            fig.canvas.manager.set_window_title(img_name)

            axes[0].imshow(img[:, :, ::-1])
            axes[0].set_title('Image')

            axes[1].imshow(shifted[:, :, ::-1])
            axes[1].set_title('Contrast +  Filter')

            axes[2].imshow(thresh, cmap='gray')
            axes[2].contour(localMax, colors='red', linewidths=1)
            axes[2].set_title('Threshold + LocalMax')

            axes[3].imshow(label2rgb(labels, image=img[:, :, ::-1]))
            axes[3].contour(labels, colors='yellow', linewidths=0.2)
            axes[3].set_title('Segmentation')

            for a in axes.ravel():
                a.set_axis_off()

            fig.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
