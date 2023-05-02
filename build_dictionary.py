import numpy as np
from glob import glob
import random
import cv2

N_FEATURES = 200
DIM_DESCRIPTOR = 128

classes = np.loadtxt('trainval/labels.csv', skiprows=1, dtype=str, delimiter=',')
all_labels = classes[:, 1].astype(np.uint8)

# create SURF object
sift = cv2.xfeatures2d.SIFT_create(nfeatures=N_FEATURES, sigma=0.4, edgeThreshold=5)

# for a training image, get keypoints and descriptors
files = glob('trainval/*/*_image.jpg')

# create a dictionary
rand_idx = random.sample(range(0, len(files)), len(files) // 10)
dict = np.zeros((N_FEATURES, DIM_DESCRIPTOR, len(files) // 10))
labels = np.zeros(len(files) // 10)
count = 0

for idx in rand_idx:

    # load the image in grayscale
    # snapshot_color = cv2.imread(snapshot)
    # cv2.imshow("colored", snapshot_color)
    # cv2.waitKey(0)
    img = cv2.imread(files[idx], cv2.IMREAD_GRAYSCALE)

    # use SURF to extract
    keypoints, descriptors = sift.detectAndCompute(img, None)
    snapshot_keys = cv2.drawKeypoints(img, keypoints, None)

    # cv2.imshow("Keypoints", snapshot_keys)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # store the descriptor into the dictionary
    print("saving features from training sample " + str(idx) + " progress " + str(count) + "/" + str(len(files) // 10))
    dict[:, :, count] = descriptors[0:N_FEATURES, :]
    labels[count] = all_labels[idx]
    count += 1

# save it into a npz file
np.savez("dictionary.npz", dict=dict, labels=labels)
