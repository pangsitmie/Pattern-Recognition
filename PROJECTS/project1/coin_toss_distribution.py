# -*- coding: utf-8 -*-
"""Coin Toss Distribution

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EkBpeTZ1DRGg3FoxigUeExvNgBQbkjgh
"""

from turtle import color
import numpy as np
from random import random
import matplotlib.pyplot as plt
import collections
import scipy.stats
import statistics
import scipy.stats as stats

"""# d = number of flips
# n = number of experiments
# prob = probability bias of the coin
# 1.	d = 100 and n =100 using a simulated coin with  = ¼ and ½.
# 2.	d = 10 and n =1000 using a simulated coin with  = 1/3 and ½.
# 3.	d = 100 and n = 10.000 using a simulated coin with  =2/5 and ½.
# 4.	d = 100 and n = 10.000 using a simulated coin with  =3/5 and ½.
"""

def generate_flip(prob, d):
    heads = []
    tails = []
    for i in range(d):
        if random() < prob:
            List1 = heads.append(1)
        else:
            List2 = tails.append(1)
    return (len(heads))

# def pdf(n,theta,x):
#     mean = n*theta
#     std = n * theta * (1- theta)
#     # mean = np.mean(x)
#     # std = np.std(x)
#     y_list = []
#     for i in range(len(x)):
#         y_list.append(1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x[i] - mean)**2 / (2 * std**2)))
#     # y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
#     print(mean, std,"...")
#     return y_list

#initialize variables
dict1 = dict()
dict2 = dict()
sum =0
d=100
n = 10000

#case1
theta1 = 1/4
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



# -------------------------------PLOTTING THE GRAPH-------------------------------
# plt.title(f'Histogram of {d} flips, {n} tries, and Theta = {theta1} & {theta2}')
# plt.xlabel("X Number of heads")
# plt.ylabel("Probability")
# plt.legend([f'Theta = {theta1}', f'Theta = {theta2}'])


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


plt.show()




# plt.bar(x,probability1)#0.5
# plt.plot(x,probability1)#0.5


# # PLOT GAUSSIAN CURVE


# fig, ax = plt.subplots(1, 1)

# #ax.scatter(od1.keys(), od1.values())
# ax.bar(list(od1.keys()),probability1)
# ax.set_xlabel('Key')
# ax.set_ylabel('Length of value')





