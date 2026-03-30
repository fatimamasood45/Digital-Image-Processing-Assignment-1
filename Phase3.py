import numpy as np
import cv2
import os
from OD_Part import Compute_OD_threshold, segment_OD
from Phase2 import Compute_OC_threshold, segment_OC

def Disc_Coeff(pred, gt):
    pred = pred > 0
    gt = gt > 0
    intersection = np.sum(pred & gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0  # so no infinite case
    disc_coef = 2 * intersection / total
    return disc_coef

if __name__ == "__main__":
    # Paths
    train_img = "G:\\Computer Engineering\\6th Semester\\DIP\\Assignment\\Drishti-GS\\Drishti-GS\\Training\\Training\\Images\\NORMAL"
    train_gt = "G:\\Computer Engineering\\6th Semester\\DIP\\Assignment\\Drishti-GS\\Drishti-GS\\Training\\Training\\GT"
    test_folder = "G:\\Computer Engineering\\6th Semester\\DIP\\Assignment\\Drishti-GS\\Drishti-GS\\Test\\Test\\Images\\normal"
    test_gt = "G:\\Computer Engineering\\6th Semester\\DIP\\Assignment\\Drishti-GS\\Drishti-GS\\Test\\Test\\Test_GT"

    # Compute thresholds from training images
    T_OD = Compute_OD_threshold(train_img, train_gt)
    T_OC = Compute_OC_threshold(train_img, train_gt)

    # Initialize for average Dice
    total_bg = 0
    total_od = 0
    total_oc = 0
    count = 0
# Run on test images
    test_images = os.listdir(test_folder)
    for im in test_images:
        img_path = os.path.join(test_folder, im)
        image = cv2.imread(img_path, 0)
        if image is None:
            continue
        img = cv2.equalizeHist(image)
        cv2.imshow("Original image", cv2.resize(image, (512, 512)))

        # Segment OD and OC
        OD_image = segment_OD(img, T_OD)
        cv2.imshow("OD image", cv2.resize(OD_image, (512, 512)))

        OC_image = segment_OC(img, OD_image, T_OC)
        cv2.imshow("OC image", cv2.resize(OC_image, (512, 512)))

        # Create final mask
        final = np.zeros(img.shape, dtype=np.uint8)
        final[OD_image > 0] = 127
        final[OC_image > 0] = 255
        cv2.imshow("Final", cv2.resize(final, (512, 512)))
        cv2.waitKey(0)

        # Load ground-truth masks
        name = os.path.splitext(im)[0]
        gt_od_path = os.path.join(test_gt, name, "SoftMap", name + "_ODsegSoftmap.png")
        gt_oc_path = os.path.join(test_gt, name, "SoftMap", name + "_cupsegSoftmap.png")
        gt_od = cv2.imread(gt_od_path, 0)
        gt_oc = cv2.imread(gt_oc_path, 0)
        if gt_od is None or gt_oc is None:
            print(f"No ground truth  for {im}")
            continue

        # Compute Dice
        d_od = Disc_Coeff(OD_image, gt_od)
        d_oc = Disc_Coeff(OC_image, gt_oc)
        bg_pred = final == 0
        bg_gt = (gt_od == 0) & (gt_oc == 0)
        d_bg = Disc_Coeff(bg_pred, bg_gt)

        # Summation for average
        total_bg += d_bg
        total_od += d_od
        total_oc += d_oc
        count += 1

        # print each image Dice coefficient
        print(f"{im} → BG Dice: {d_bg:.2f},   OD Dice: {d_od:.2f},   OC Dice: {d_oc:.2f}")

    # print average Dice
    avg_bg = total_bg / count
    avg_od = total_od / count
    avg_oc = total_oc / count
    print("\nAverage Dice of Test Images:")
    print(f"BG: {avg_bg*100:.2f}%,   OD: {avg_od*100:.2f}%,   OC: {avg_oc*100:.2f}%")

