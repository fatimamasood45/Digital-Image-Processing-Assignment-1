# 👁️ Retinal Fundus Image Segmentation

### Optic Disc (OD) & Optic Cup (OC) Detection using Connected Component Labeling

---

## 📌 Overview

This project focuses on **segmenting retinal fundus images** to detect the **Optic Disc (OD)** and **Optic Cup (OC)** using **Connected Component Labeling (CCL)**.

The implementation is done in **Python (OpenCV + NumPy)** and follows a **multi-phase approach** involving thresholding, segmentation, and evaluation using the **Dice Coefficient**.

---

## 🎯 Objectives

* Segment **Optic Disc (OD)** from retinal images
* Segment **Optic Cup (OC)** within OD region
* Apply **8-connected component labeling**
* Generate a **3-class segmentation mask**:

  * Background → `0`
  * OD → `127`
  * OC → `255`
* Evaluate performance using **Dice Coefficient**

---

## 🧠 Methodology

### 🔹 Phase 1: Optic Disc (OD) Segmentation

* Convert image to grayscale
* Apply **Histogram Equalization** (contrast enhancement)
* Extract OD pixels from ground truth masks
* Compute threshold using **28th percentile**
* Binarize image
* Apply **8-connected component labeling**
* Select **largest connected component** as OD

---

### 🔹 Phase 2: Optic Cup (OC) Segmentation

* Work inside detected OD region
* Extract OC pixels from training masks
* Compute threshold using **65th percentile**
* Binarize OD region
* Apply **CCL**
* Select **largest component** as OC
* Combine OD & OC into final mask

---

### 🔹 Phase 3: Evaluation (Dice Coefficient)

* Compare predicted masks with ground truth
* Compute Dice for:

  * Background
  * Optic Disc
  * Optic Cup

**Formula:**

```
Dice = 2 × |Intersection| / (|Prediction| + |Ground Truth|)
```

---

## ⚙️ Implementation Details

### 🔸 1. Connected Component Labeling (CCL)

* Custom implementation of **8-connected labeling**
* Two-pass algorithm:

  * First pass → assign labels + equivalence
  * Second pass → resolve equivalences

---

### 🔸 2. Threshold Computation

* Uses training dataset
* Percentile-based thresholding:

  * OD → 28%
  * OC → 65%

---

### 🔸 3. Segmentation Pipeline

* Preprocessing → Thresholding → CCL → Largest Component Selection

---

### 🔸 4. Dice Coefficient

* Measures overlap between prediction & ground truth
* Used for performance evaluation

---

## 🗂️ Project Structure

```
├── main.py                # 8-connected component labeling
├── OD_Part.py            # Optic Disc segmentation
├── Phase2.py             # Optic Cup segmentation
├── Dice.py               # Dice coefficient calculation
├── dataset/              # Drishti-GS dataset
└── README.md
```

---

## 📊 Dataset

* **Drishti-GS Dataset**
* Contains:

  * Training images
  * Ground truth masks (OD & OC)
  * Test images

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install numpy opencv-python
```

---

### 2️⃣ Update Paths

Modify dataset paths in code:

```python
train_img = "path_to_training_images"
train_gt = "path_to_ground_truth"
test_folder = "path_to_test_images"
```

---

## 📈 Results

* Successfully segmented:

  * Optic Disc (OD)
  * Optic Cup (OC)
* Generated **3-class masks**
* Evaluated using Dice coefficient for:

  * Background
  * OD
  * OC

Example outputs include:

* OD segmented images
* OC segmented images
* Final combined masks
* Dice scores for each test image

---

## ⚠️ Challenges & Solutions

### 🔹 Challenge: Noise in Thresholding

**Solution:** Percentile-based thresholding instead of fixed value

---

### 🔹 Challenge: Multiple Components

**Solution:** Select **largest connected component**

---

### 🔹 Challenge: OC Detection Accuracy

**Solution:** Restrict segmentation inside OD region

---

### 🔹 Challenge: Missing Ground Truth

**Solution:** Skip invalid samples during evaluation

---

## 🛠️ Technologies Used

* **Python**
* **OpenCV**
* **NumPy**
* Image Processing Techniques:

  * Histogram Equalization
  * Thresholding
  * Connected Component Labeling

