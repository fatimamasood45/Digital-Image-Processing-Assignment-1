import numpy as np
import cv2

def eight_connected_components(image):
    rows, cols = image.shape
    output = np.zeros((rows, cols), dtype=np.uint16)
    count = 0
    equivalence = {}

    # First Pass
    for i in range(rows):
        for j in range(cols):

            if image[i, j] !=0:

                if i >= 1:
                    top = output[i-1, j]
                else:
                    top = 0

                if j >= 1:
                    left = output[i, j-1]
                else:
                    left = 0

                if i > 0 and j > 0:
                    top_left = output[i-1, j-1]
                else:
                    top_left = 0

                if i > 0 and j < cols-1:
                    top_right = output[i-1, j+1]
                else:
                    top_right = 0

                # Case 1: No neighbors
                if top == 0 and left == 0 and top_left == 0 and top_right == 0:
                    count += 1
                    output[i, j] = count
                    equivalence[count] = count

                else:
                    # Finding minimum non-zero neighbor
                    min_label = 0

                    neighbors = [left, top, top_left, top_right]

                    for n in neighbors:
                        if n != 0:
                            if min_label == 0:
                                min_label = n
                            elif n < min_label:
                                min_label = n

                    output[i, j] = min_label

                    # Storing in equivalency list
                    for n in neighbors:
                        if n != 0 and n != min_label:
                            equivalence[n] = min_label

    # Second Pass
    for i in range(rows):
        for j in range(cols):
            if output[i, j] != 0:
                label = output[i, j]

                while equivalence[label] != label:
                    label = equivalence[label]

                output[i, j] = label

    # Count objects
    objects = output[output > 0]
    unique_objects = np.unique(objects)
    Total_objects = len(unique_objects)

    return output


if __name__ == '__main__':
    image = cv2.imread("x_image.png", 0)
    labeled_image = eight_connected_components(image)