import numpy as np
import cv2
import os
from main import eight_connected_components

# Taking threshold from training images
def Compute_OD_threshold(folder_path, mask_folder):
    all_pixels = []
    images = os.listdir(folder_path)

    for im in images:
        img = cv2.imread(os.path.join(folder_path, im), 0)
        img = cv2.equalizeHist(img)

        name = os.path.splitext(im)[0]
        mask_path = os.path.join(mask_folder, name, "SoftMap", name + "_ODsegSoftmap.png")
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            continue

        OD_pixels = img[mask > 0]
        all_pixels.extend(OD_pixels)

    all_pixels = np.array(all_pixels)
    # Using percentile to avoid noise
    T = np.percentile(all_pixels, 29)
    print("OD Threshold:", T)
    return int(T)


# Segment OD from image
def segment_OD(image, T):
    binary = np.zeros(image.shape,dtype=np.uint8)
    binary[image >= T] = 255

    labels = eight_connected_components(binary)
    # find largest component
    unique, counts = np.unique(labels[labels > 0], return_counts=True)
    largest = unique[np.argmax(counts)]

    OD_mask = np.zeros(image.shape,dtype=np.uint8)
    OD_mask[labels == largest] = 255

    return OD_mask

if __name__ == "__main__":
    train_img = "G:\\Computer Engineering\\6th Semester\\DIP\\Assignment\\Drishti-GS\\Drishti-GS\\Training\\Training\\Images\\NORMAL"
    train_gt = "G:\\Computer Engineering\\6th Semester\\DIP\\Assignment\\Drishti-GS\\Drishti-GS\\Training\\Training\\GT"
    test_folder = "G:\\Computer Engineering\\6th Semester\\DIP\\Assignment\\Drishti-GS\\Drishti-GS\\Test\\Test\\Images\\normal"

# OD V-set threshold
    T = Compute_OD_threshold(train_img, train_gt)

# Run on test images
    for im in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder, im), 0)
        img = cv2.equalizeHist(img)

        OD = segment_OD(img, T)

        cv2.imshow("Original", cv2.resize(img,(512,512)))
        cv2.imshow("OD Segmented", cv2.resize(OD,(512,512)))
        cv2.waitKey(0)


