import numpy as np
from random import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


# d = number of flips
# n = number of experiments
# prob = probability bias of the coin
# 1.	d = 100 and n =100 using a simulated coin with  = ¼ and ½.
# 2.	d = 10 and n =1000 using a simulated coin with  = 1/3 and ½.
# 3.	d = 100 and n = 10000 using a simulated coin with  =2/5 and ½.
# 4.	d = 100 and n = 10.000 using a simulated coin with  =3/5 and ½.


def generate_flip(prob, d):
    heads = []
    tails = []
    for i in range(d):
        if random() < prob:
            List1 = heads.append(1)
        else:
            List2 = tails.append(1)
    return (len(heads))


# initialize variables
dict1 = dict()
dict2 = dict()
sum = 0
d = 100
n = 10000

# case1
theta1 = 3/5
print("#(head, tails) for", d, "flips and", n, "tries with theta=", theta1)
for times in range(0, n):
    z = generate_flip(theta1, d)
    if z not in dict1:
        dict1[z] = 1
    else:
        dict1[z] = dict1[z]+1
print("The probability of (Head, Tail)>>>")
for key in dict1:
    print(key, " Head: ", dict1[key]/(n))
    sum += (dict1[key]/(n))
print("The sum of the probabilities is:", sum, '\n')

# case2
theta2 = 1/2
sum = 0
print("#(head, tails) for", d, "flips and", n, "tries with theta=", theta2)
for times in range(0, n):
    z = generate_flip(theta2, d)
    if z not in dict2:
        dict2[z] = 1
    else:
        dict2[z] = dict2[z]+1
print("The probability of (Head, Tail)>>>")
for key in dict2:
    print(key, " Head: ", dict2[key]/(n))
    sum += (dict2[key]/(n))
print("The sum of the probabilities is:", sum, '\n')


lists = sorted(dict1.items())  # sorted by key, return a list of tuples
x, y = zip(*lists)  # unpack a list of pairs into two tuples

lists = sorted(dict2.items())  # sorted by key, return a list of tuples
v, w = zip(*lists)  # unpack a list of pairs into two tuples

plt.bar(x, y)  # 0.5
plt.plot(x, y)  # 0.5

plt.title(
    f'Histogram of {d} flips, {n} tries, and Theta = {theta1} & {theta2}')
plt.xlabel("X Number of heads")
plt.ylabel("Frequency")
plt.legend([f'Theta = {theta1}', f'Theta = {theta2}'])
plt.show()

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()