import numpy as np
from random import random
import matplotlib.pyplot as plt
import collections


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

def pdf(n,theta,x):
    mean = n*theta
    std = n * theta * (1- theta)
    # mean = np.mean(x)
    # std = np.std(x)
    y_list = []
    for i in range(x):
        y_list.append(1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2)))
    # y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
    print(mean, std,"...")
    return y_list



#initialize variables
dict1 = dict()
dict2 = dict()
sum =0
d=100
n = 10000

#case1
theta1 = 1/2
print("#(head, tails) for", d, "flips and", n ,"tries with theta=" ,theta1)
for times in range (0,n):
        z = generate_flip(theta1,d)
        if z not in dict1:
            dict1[z] = 1
        else:
            dict1[z] =dict1[z]+1
print("The probability of (Head, Tail)>>>")
for key in dict1:
    print(key," Head: ", dict1[key]/(n))
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
    print(key," Head: ", dict2[key]/(n))
    sum += (dict2[key]/(n))
print("The sum of the probabilities is:", sum, '\n')



od1 = collections.OrderedDict(sorted(dict1.items()))
print(od1)

od2 = collections.OrderedDict(sorted(dict2.items()))
print(od2)


lists = sorted(od1.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples

print(x)

lists = sorted(od2.items()) # sorted by key, return a list of tuples
v, w = zip(*lists) # unpack a list of pairs into two tuples


# plt.figure(figsize = (6, 6))

#plot data

output1 = list(od1.values())
probability1 = [i / n for i in output1]
print("all values in dict1 are:", probability1)

output2 = list(od2.values())
probability2 = [i / n for i in output2]
print("all values in dict2 are:", probability2)


# plt.bar(x,probability1)#0.5
# plt.plot(x,probability1)#0.5

print(output1)

print("mean:", np.mean(output1))
# PLOT THE GAUSIAN CURVE BASED ON PDF
gaus1 = pdf(n,theta1,probability1)
print("gaus1: ",gaus1)
# plt.plot(x, gaus1, color = 'black', linestyle = 'dashed')
# plt.scatter( x, gaus1, marker = 'o', s = 25, color = 'red')



# plt.bar(v,w)#0.5
# plt.plot(v,w)#0.5



plt.title(f'Histogram of {d} flips, {n} tries, and Theta = {theta1} & {theta2}')
plt.xlabel("X Number of heads")
plt.ylabel("Probability")
plt.legend([f'Theta = {theta1}', f'Theta = {theta2}'])


plt.show()

