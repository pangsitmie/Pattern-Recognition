#JERIEL B10817055
from types import new_class
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.cluster import KMeans
# from sklearn.svm import SVC
# from scipy.spatial.distance import pdist
from multiprocessing import Process, Pool, cpu_count, Queue, Manager

# LOAD DATA
def load_all_ground_truth():
    b1 = []
    b2 = []
    b3 = []

    for index in range(11, 29):
        path = r"C:\Users\jerie\Documents\GitHub\Pattern-Recognition\PROJECTS\project_4\npy\gtnpy\slices_" + \
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


def load_all_data():
    data = np.load(
        r"C:\Users\jerie\Documents\GitHub\Pattern-Recognition\PROJECTS\project_4\npy\datanpy\simulation_Brain_Web_noise3_rf0_5mm_all_images.npy", allow_pickle=True).item()

    pd = data['AI_PD_n3_rf0'].swapaxes(0, 2).swapaxes(1, 2)
    t1 = data['AI_T1_n3_rf0'].swapaxes(0, 2).swapaxes(1, 2)
    t2 = data['AI_T2_n3_rf0'].swapaxes(0, 2).swapaxes(1, 2)

    # plt.imshow(t1[1])

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

        label = np.zeros(39277)
        for index, i in enumerate(range(39277*(pic_num), 39277*(pic_num+1))):
            current_point = x[i, :]
            d1 = np.sqrt(np.sum(np.square(c1 - current_point)))
            d2 = np.sqrt(np.sum(np.square(c2 - current_point)))
            d3 = np.sqrt(np.sum(np.square(c3 - current_point)))
            d4 = np.sqrt(np.sum(np.square(c4 - current_point)))
            min_value = min([d1, d2, d3, d4])
            min_index = [d1, d2, d3, d4].index(min_value)

            if min_index == 0:
                group1.append(current_point)
                label[index] = 0
            elif min_index == 1:
                group2.append(current_point)
                label[index] = 1
            elif min_index == 2:
                group3.append(current_point)
                label[index] = 2
            elif min_index == 3:
                group4.append(current_point)
                label[index] = 3

        temp_c = np.round(np.array([c1, c2, c3, c4]), 3)

        c1 = np.mean(np.array(group1), axis=0)
        c2 = np.mean(np.array(group2), axis=0)
        c3 = np.mean(np.array(group3), axis=0)
        c4 = np.mean(np.array(group4), axis=0)

        now_c = np.round(np.array([c1, c2, c3, c4]), 3)
        s_c1 = str(c1)
        s_c2 = str(c2)
        s_c3 = str(c3)
        s_c4 = str(c4)

        title = 'c1: ' + s_c1 + '       ' + 'c2: ' + s_c2 + \
            '       ' 'c3: ' + s_c3 + '     ' + 'c4: ' + s_c4
        if(now_c == temp_c).all():
            break
        label = np.asarray(label).reshape(181, 217)
        label1 = np.copy(label)
        label1[label1 != 1] = 0

        label2 = np.copy(label)
        label2[label2 != 2] = 0

        label3 = np.copy(label)
        label3[label3 != 3] = 0
        manager = plt.get_current_fig_manager()

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.subplot(2, 4, 1)
        plt.gca().set_title(title, loc='left')

        plt.imshow(label)
        plt.subplot(2, 4, 2)
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
    b, b1, b2, b3 = load_all_ground_truth()
    pd, t1, t2 = load_all_data()
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
    self_Kmeans(x, 1)
    # plt.show()
