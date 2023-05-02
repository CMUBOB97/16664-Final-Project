import numpy as np
from glob import glob
import csv
import cv2

N_FEATURES = 200
DIM_DESCRIPTOR = 128

def chi_2_dist(A, B):

    numerator = (A - B) ** 2
    denominator = (A + B)

    return np.sum(numerator)/np.sum(denominator)

def sift_predict(path, sift, map):
    files = glob('{}/*/*_image.jpg'.format(path))
    files.sort()

    name = '{}/prediction.csv'.format(path)

    # sift descriptors and labels
    train_descriptors = map['dict']
    train_labels = map['labels']
    train_len = np.size(train_labels)
    count = 1

    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image', 'label'])

        for file in files:
            guid = file.split('\\')[-2]
            idx = file.split('\\')[-1].replace('_image.jpg', '')

            # use sift detector to extract descriptors
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = sift.detectAndCompute(img, None)
            top_4_dist = -1 * np.ones(4)
            top_4_labels = -1 * np.ones(4)

            # compare to the dictionary and get 4 closest ones
            for cmp_idx in range(0, train_len):
                dist = chi_2_dist(descriptors[0:N_FEATURES, :], train_descriptors[:, :, cmp_idx])
                label = train_labels[cmp_idx]
                if cmp_idx < 4:
                    top_4_dist[cmp_idx] = dist
                    top_4_labels[cmp_idx] = label
                else:
                    for option in range(0, 4):
                        if dist < top_4_dist[option]:
                            top_4_dist[option] = dist
                            top_4_labels[option] = label

            # count labels
            label_0_count = np.sum(top_4_labels == 0)
            label_1_count = np.sum(top_4_labels == 1)
            label_2_count = np.sum(top_4_labels == 2)

            # choose the max
            label = np.argmax([label_0_count, label_1_count, label_2_count])

            writer.writerow(['{}/{}'.format(guid, idx), label])
            print("write prediction for " + file + " progress " + str(count) + "/" + str(len(files)))
            count += 1

    print('Wrote report file `{}`'.format(name))

if __name__ == "__main__":

    # sift extractor
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=N_FEATURES, sigma=0.4, edgeThreshold=5)

    # load sift map from training data
    map = np.load("dictionary.npz")

    for path in ['test']:
        sift_predict(path, sift, map)