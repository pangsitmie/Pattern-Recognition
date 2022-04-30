import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def func(d, n, p):
    re = [[], []]
    for j in range(len(p)):
        for i in range(n):
            re[j].append(Counter(np.random.choice(
                a=[1, 0], size=d, p=[p[j], 1-p[j]]))[1])
    return re


def rnd(data):
    c = np.array([0.0, 0.0])
    while c[0] == c[1]:
        c = np.random.randint(min(data), max(data) + 1, 2)
    return c


def grouping(data, c):
    re = [[], []]
    for i in data:
        diff1 = abs(i - c[0])
        diff2 = abs(i - c[1])
        if diff1 == diff2:
            re[0].append(i)
            re[1].append(i)
        elif diff1 < diff2:
            re[0].append(i)
        else:
            re[1].append(i)

    return re


def meaning(group):
    re = np.array([0.0, 0.0])
    for i in range(len(group)):
        sum = 0
        for j in group[i]:
            sum += j
        re[i] = (sum / len(group[i]))
    return re


d = 100
n = 10000
p = [0.6, 0.5]
g = []
temp = 0
line = []

lists = func(d, n, p)

print(lists[0])
print(lists[1])


centers = np.array([[0.0, 0.0], [0.0, 0.0]])
for i in range(2):
    centers[i] = rnd(lists[i])

print(centers)

while True:
    for i in range(2):
        g.append(grouping(lists[i], centers[i]))
        centers[i] = meaning(g[i])
        print(centers)
        line.append((centers[i][0] + centers[i][1]) / 2)
        print(line)
        plt.hist(lists[i], bins=50)
        plt.axvline(line[i], 0.0, 1.0, linestyle='-', color='black')
        print(type(lists[i]))
    plt.legend()
    plt.pause(0.5)

    plt.cla()
    plt.clf()
    line.clear()
    g.clear()
plt.show()
