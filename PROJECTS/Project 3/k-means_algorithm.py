# -*- coding: utf-8 -*-
"""Coin Toss Distribution

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EkBpeTZ1DRGg3FoxigUeExvNgBQbkjgh
"""
from optparse import Values
import random as rand
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy.stats as stats
from matplotlib.animation import FuncAnimation
from collections import defaultdict


"""# d = number of flips
# n = number of experiments
# prob = probability bias of the coin
# 1.	d = 100 and n =100 using a simulated coin with  = ¼ and ½.
# 2.	d = 10 and n =1000 using a simulated coin with  = 1/3 and ½.
# 3.	d = 100 and n = 10.000 using a simulated coin with  =2/5 and ½.
# 4.	d = 100 and n = 10.000 using a simulated coin with  =3/5 and ½.
"""


#initialize variables
dict1 = dict()
dict2 = dict()
sum =0
d=500
n = 10000

centeroid = []



# ------------------------------FUNCTIONS--------------------------
def generate_flip(prob, d):
    heads = []
    tails = []
    for i in range(d):
        if rand.random() < prob:
            List1 = heads.append(1)
        else:
            List2 = tails.append(1)
    return (len(heads))

def k_means(dataset,k):

    key_list = list(dataset)
    size = len(key_list)


    random_list = []
    for i in range (0,k):
        rand_idx = rand.randint(int(size/4), int(3*size/4))
        print("RAND IDX: ", rand_idx)
        centeroid.append(key_list[rand_idx])
        print("KEY IDX[", rand_idx,"]: ",centeroid[i])
        random_list.append(rand_idx)
    
    print("Random list: ", random_list)

def distance(x1,x2):
    return (abs(x1-x2))

# def Average(lst):
#     return sum(lst) / len(lst)

# def animate(i):
#     x_cals.append(next(index))
#     y_vals.append(random.randint(0.5))

# ani = FuncAnimation(plt.gcf(), animate, interval = 1000)

    

    



#case1
theta1 = 2/5
print("#(head, tails) for", d, "flips and", n ,"tries with theta=" ,theta1)
for times in range (0,n):
        z = generate_flip(theta1,d)
        if z not in dict1:
            dict1[z] = 1
        else:
            dict1[z] =dict1[z]+1
print("The probability of (Head, Tail)>>>")
for key in dict1:
    #print(key," Head: ", dict1[key]/(n))
    sum += (dict1[key]/(n))
print("The sum of the probabilities is:", sum, '\n')

#case2
theta2 = 1/2
sum=0
print("#(head, tails) for", d, "flips and", n ,"tries with theta=" ,theta2)
for times in range (0,n):
        z = generate_flip(theta2,d)
        if z not in dict2:
            dict2[z] = 1
        else:
            dict2[z] =dict2[z]+1
print("The probability of (Head, Tail)>>>")
for key in dict2:
    # print(key," Head: ", dict2[key]/(n))
    sum += (dict2[key]/(n))
print("The sum of the probabilities is:", sum, '\n')

# CASE1 ORDERED DICT (SORTED BAES ON KEYS)
od1 = collections.OrderedDict(sorted(dict1.items()))
print(od1)

# CASE2 ORDERED DICT (SORTED BAES ON KEYS)
od2 = collections.OrderedDict(sorted(dict2.items()))
print(od2)


lists = sorted(od1.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples

lists = sorted(od2.items()) # sorted by key, return a list of tuples
v, w = zip(*lists) # unpack a list of pairs into two tuples


output1 = list(od1.values())
probability1 = [i / n for i in output1]
print("sorted x keys of dict1 are:", list(od1.keys()))
print("sorted x values of dict1 are:", output1)
print("all probability in dict1 are:", probability1)

print('\n')
output2 = list(od2.values())
probability2 = [i / n for i in output2]
print("sorted x keys of dict1 are:", od2.keys())
print("sorted x values of dict1 are:", output2)
print("all probability in dict2 are:", probability2)



d3 = collections.OrderedDict(list(od1.items()) + list(od2.items()))
od3 = collections.OrderedDict(sorted(d3.items()))

print("od1: ",od1)
print("od2: ",od2)
print("od3: ",od3)


print("len od od3 ", len(od3))
k_means(od3,2)















# -------------------------------PLOTTING THE GRAPH-------------------------------

fig, ax = plt.subplots()

# plot the 1st theta and gausian
l_list1 = [k for k, v in od1.items() for _ in range(v)]
mu1 = np.mean(l_list1)
sigma1 = np.std(l_list1)
plt.bar(list(od1.keys()),probability1)
u = np.linspace(mu1 - 4 * sigma1, mu1 + 4 * sigma1, 100)
ax = plt.twinx()



# plot the 2st theta and gausian
l_list2 = [k for k, v in od2.items() for _ in range(v)]
mu2 = np.mean(l_list2)
sigma2 = np.std(l_list2)
plt.bar(list(od2.keys()),probability2, color='orange')
u2 = np.linspace(mu2 - 4 * sigma2, mu2 + 4 * sigma2, 100)
#ax2 = plt.twinx()

ax.plot(u, stats.norm.pdf(u, mu1, sigma1), color='purple')
ax.plot(u2, stats.norm.pdf(u2, mu2, sigma2), color='crimson')


ax.set_title(f'Histogram of {d} flips, {n} tries, and Theta = {theta1} & {theta2}')
fig.supxlabel('X Number of heads')
fig.supylabel('Probability')


# plt.axvline(x = 40)
# plt.axvline(x = 50)
for i in range (0,len(centeroid)):
    print(centeroid[i])
    plt.axvline(x = centeroid[i], ls='--')

middle = (centeroid[0]+centeroid[1])/2
plt.axvline(x= middle, color='r')
    

# --------------------------------
all_key_list = []
key_list = list(od1)
for i in range(0, len(od1)):
    all_key_list.append(key_list[i])

keylist2 = list(od2)
for i in range(0, len(od2)):
    all_key_list.append(keylist2[i])
print("All keys: ", all_key_list)

cluster1 = []
cluster2 = []
for i in range(0,100):
    for point in all_key_list:
        x=[]
        for idx in range(0,len(centeroid)):
            x.append(distance(centeroid[idx], point))
        # print("X: ",x)

        if(x[0] < x[1]):
            cluster1.append(point)
        else:
            cluster2.append(point)

    AVG_cluster1 = np.average(cluster1)
    AVG_cluster2 = np.average(cluster2)
    print("cluster1: ", cluster1, "AVG cluster1: ",AVG_cluster1)
    print("cluster2: ", cluster2, "AVG cluster2: ",AVG_cluster2)

    centeroid[0] = AVG_cluster1
    centeroid[1] = AVG_cluster2
    od1 = cluster1
    od2 = cluster2

    cluster1.clear()
    cluster2.clear()

for i in range (0,len(centeroid)):
    plt.axvline(x = centeroid[i], ls='--', color = 'black')
# ------------------------------------


    



        
# for i in range (0,1):
#     center1_1 = [distance(centeroid[0], point) for point in od3]
#     center1_2 = [distance(centeroid[1], point) for point in od3]
    

#     print(center1_1)
#     print(center1_2)
#     center1_final = center1_1 + center1_2
#     print(center1_final)

#     avg_center1 = np.average(center1_final)
#     print("Average of the list0 =", round(avg_center1, 2))



    
#     center2_1 = [distance(centeroid[1], point) for point in od1]
#     center2_2 = [distance(centeroid[1], point) for point in od2]

#     print(center2_1)
#     print(center2_2)
#     center2_final = center2_1 + center2_2
#     print(center2_final)

#     avg_center2 = np.average(center2_final)
#     print("Average of the list1 =", round(avg_center2, 2))

#     centeroid[0] = avg_center1
#     centeroid[1] = avg_center2



plt.show()



