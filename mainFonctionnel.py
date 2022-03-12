from os import listdir
from scipy import ndimage
from skimage.feature import peak_local_max
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, morphology, segmentation
from skimage.segmentation import watershed
from skimage import measure
import pandas as pd


def load_img():
    """
    Description :
    Méthode pour le chargement des images.
    """

    img_list = {}

    for img in listdir('./Images/'):
        img_list[img] = cv2.imread('./Images/' + img, type=np.uint)
        # img_list[img] = cv2.resize(img_list[img], DIM_IMG)

    return img_list


def save_df(df, img_name):
    """
    Description :
    Méthode pour la sauvegarde des dataframes au format csv
    """

    df.to_csv("./Output/Dataframes/" + img_name.split(".")[0] + '.csv', index=False, header=True)


def save_results(img, img_name):
    """
    Description :
    Méthode pour la sauvegarde des images segmentées
    """

    plt.imsave("./Output/Images/" + img_name, img)


def display_labels(labels, img):
    """
    Description :
    Méthode pour l'affichage des labels.
    """
    for i, segVal in enumerate(np.unique(labels)):
        print("[x] inspecting segment %d" % i)

        mask = np.zeros(img.shape[:2], dtype="uint8")
        mask[labels == segVal] = 255

        cv2.imshow("Applied", cv2.bitwise_and(img, img, mask=mask))
        cv2.waitKey(0)



def main():
    # params
    display = False

    img_list = load_img()

    for img_name, img in img_list.items():

        # Contrast stretching
        p2, p98 = np.percentile(img, (2, 98))
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

        # Adaptive Equalization
        img_eq = exposure.equalize_adapthist(img, clip_limit=0.03)

        # mean shift
        shifted = cv2.pyrMeanShiftFiltering(img, 21, 31)

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

        # rgb mean
        regions = measure.regionprops(labels, intensity_image=img)
        data = pd.DataFrame(columns=["Grain isolé n°", "Moyenne de B", "Moyenne de G", "Moyenne de R"])
        for i in range(len(np.unique(labels)) - 1):
            r = regions[i]
            data.loc[i] = ['Grain isolé n°' + str(i + 1)] \
                          + [round(r.intensity_mean[0], 2)] \
                          + [round(r.intensity_mean[1], 2)] \
                          + [round(r.intensity_mean[2], 2)]

        if display:
            display_labels(labels, img)

        # display
        fig, axes = plt.subplots(ncols=2, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(img, cmap=plt.cm.gray)
        ax[0].set_title('Image')

        #ax[1].imshow(labels, cmap=plt.cm.tab20b)
        ax[1].imshow(img)
        ax[1].contour(labels, colors='yellow', linewidths=0.5)
        ax[1].set_title('Segmentation')


        for a in ax:
            a.set_axis_off()

        fig.tight_layout()
        plt.show()

        save_df(data, img_name)
        save_results(segmentation.mark_boundaries(img, labels), img_name)





if __name__ == "__main__":
    main()






