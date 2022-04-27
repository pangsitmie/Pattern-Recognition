import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from scipy.spatial.distance import pdist
from multiprocessing import Process, Pool, cpu_count, Queue, Manager

# LOAD DATA


def all_ground_truth():
    b1 = []
    b2 = []
    b3 = []

    for index in range(11, 29):
        path = r"C:\Users\pangsitmie\Documents\GitHub\Pattern-Recognition\PROJECTS\project_4\npy\gtnpy\slices_" + \
            str(index)+"_groundtruth.npy"
        gt = np.load(path, allow_pickle=True).item()
        b1.append(gt['B1'])
        b2.append(gt['B2'])
        b3.append(gt['B3'])

    b1 = np.array(b1)
    b2 = np.array(b2)
    b3 = np.array(b3)

    b1 = np.where(b1 == 0.6, 1, b1)
    b2 = np.where(b2 == 0.6, 2, b2)
    b3 = np.where(b3 == 0.6, 3, b3)

    # plt.imshow(b3[0])

    b = b1+b2+b3
    return b, b1, b2, b3


def all_data():
    data = np.load(
        r"C:\Users\pangsitmie\Documents\GitHub\Pattern-Recognition\PROJECTS\project_4\npy\datanpy\simulation_Brain_Web_noise3_rf0_5mm_all_images.npy", allow_pickle=True).item()

    pd = data['AI_PD_n3_rf0'].swapaxes(0, 2).swapaxes(1, 2)
    t1 = data['AI_T1_n3_rf0'].swapaxes(0, 2).swapaxes(1, 2)
    t2 = data['AI_T2_n3_rf0'].swapaxes(0, 2).swapaxes(1, 2)

    plt.imshow(t1[1])

    return pd, t1, t2


def self_Kmeans(x, pic_num):
    c1 = np.array([10, 10, 10])
    c2 = np.array([200, 50, 100])
    c3 = np.array([100, 30, 100])
    c4 = np.array([200, 100, 50])

    while(True):
        group1 = []
        group2 = []
        group3 = []
        group4 = []

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.subplot(2, 4, 1)
    plt.gca().set_title(title, loc='left')

    plt.imshow(label)
    plt.subplot(3, 4, 2)

    plt.imshow(label1)
    plt.subplot(2, 4, 3)

    plt.imshow(label2)
    plt.subplot(2, 4, 4)

    plt.imshow(label3)

    plt.subplot(2, 4, 5)
    plt.imshow(b[pic_num, ...])

    plt.subplot(2, 4, 6)
    plt.imshow(b1[pic_num, ...])

    plt.subplot(2, 4, 7)
    plt.imshow(b2[pic_num, ...])

    plt.subplot(2, 4, 8)
    plt.imshow(b3[pic_num, ...])

    plt.pause(0.01)


if __name__ == "__main__":
    # data import
    b, b1, b2, b3 = all_ground_truth()
    pd, t1, t2 = all_data()
    mask = np.zeros((18, 181, 217))
    mask[b > 0] = 1
    pd = pd*mask
    t1 = t1*mask
    t2 = t2*mask
    plt.imshow(b[0])
    pd = pd.reshape(181*217*18, 1)
    t1 = t1.reshape(181*217*18, 1)
    t2 = t2.reshape(181*217*18, 1)

    x = np.hstack((pd, t1, t2))
    print(t1[0])
    # self_Kmeans(x,1)
    plt.show()
