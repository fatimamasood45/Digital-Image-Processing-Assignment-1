import numpy as np
import cv2
import os
from OD_Part import Compute_OD_threshold, segment_OD
from main import eight_connected_components

def Compute_OC_threshold(folder_path, mask_folder):
    all_pixels = []
    images = os.listdir(folder_path)

    for im in images:
        img = cv2.imread(os.path.join(folder_path, im), 0)
        img = cv2.equalizeHist(img)

        name = os.path.splitext(im)[0]
        cup_mask_path = os.path.join(mask_folder, name, "SoftMap", name + "_cupsegSoftmap.png")
        mask = cv2.imread(cup_mask_path, 0)

        if img is None or mask is None:
            continue

        OC_pixels = img[mask > 0]
        all_pixels.extend(OC_pixels)

    all_pixels = np.array(all_pixels)
    T = np.percentile(all_pixels, 31)
    print("OC Threshold:", T)
    return int(T)

def segment_OC(image, OD_mask, T_OC):
    binary = np.zeros(image.shape,dtype=np.uint8)
    binary[(image >= T_OC) & (OD_mask > 0)] = 255

    labels = eight_connected_components(binary)
    objs = labels[labels > 0]

    if len(objs)==0:
        return np.zeros(image.shape,dtype=np.uint8)

    unique, counts = np.unique(objs, return_counts=True)
    largest = unique[np.argmax(counts)]

    OC_mask = np.zeros(image.shape,dtype=np.uint8)
    OC_mask[labels == largest] = 255

    return OC_mask

if __name__ == "__main__":
    train_img = "G:\\Computer Engineering\\6th Semester\\DIP\\Assignment\\Drishti-GS\\Drishti-GS\\Training\\Training\\Images\\NORMAL"
    train_gt = "G:\\Computer Engineering\\6th Semester\\DIP\\Assignment\\Drishti-GS\\Drishti-GS\\Training\\Training\\GT"
    test_folder = "G:\\Computer Engineering\\6th Semester\\DIP\\Assignment\\Drishti-GS\\Drishti-GS\\Test\\Test\\Images\\normal"

    T_OD = Compute_OD_threshold(train_img, train_gt)
    T_OC = Compute_OC_threshold(train_img, train_gt)

# Run on test images
    test_images = os.listdir(test_folder)
    for im in test_images:
        img = cv2.imread(os.path.join(test_folder, im), 0)
        img = cv2.equalizeHist(img)

        OD = segment_OD(img, T_OD)
        cv2.imshow("OD image", cv2.resize(OD, (512, 512)))

        OC = segment_OC(img, OD, T_OC)
        cv2.imshow("OC image", cv2.resize(OC, (512, 512)))

        final = np.zeros(img.shape,dtype=np.uint8)
        final[OD > 0] = 127
        final[OC > 0] = 255

        cv2.imshow("Final", cv2.resize(final, (512, 512)))
        cv2.waitKey(0)
