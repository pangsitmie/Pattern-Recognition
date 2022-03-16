import numpy as np
from random import random
import matplotlib.pyplot as plt


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
    return (len(heads), len(tails))


# main
# d = int(input("Enter number of time to flip the coin: "))
# n = int(input("Enter how may tries to conduct the experiment: "))
# prob = float(input("Enter coin probability: "))

#print(generate_flip(prob,d))

dict1 = dict()
sum =0

# d=100, n = 100, Theta = 1/4
# print("#(head, tails) for 100 flips and 100 tries with theta=1/4")
# for times in range (0,n):
#         z = generate_flip(prob,d)
#         if z not in dict1:
#             dict1[z] = 1
#         else:
#             dict1[z] =dict1[z]+1
# print(dict1)
# print("\nFor the probability of heads to tails>>>")
# for key in dict1:
#     print(key, dict1[key]/(n))
#     sum += (dict1[key]/(n))
# print("The sum of the probabilities is:", sum, '\n')


# d = 100 and n =100 using a simulated coin with theta = 0.25
d=100
n = 100
theta = 0.25
print("#(head, tails) for", d, "flips and", n ,"tries with theta=" ,theta)
for times in range (0,n):
        z = generate_flip(theta,d)
        if z not in dict1:
            dict1[z] = 1
        else:
            dict1[z] =dict1[z]+1
print("The probability of (Head, Tail)>>>")
for key in dict1:
    print(key, dict1[key]/(n))
    sum += (dict1[key]/(n))
print("The sum of the probabilities is:", sum, '\n')



# d = 100 and n =100 using a simulated coin with theta = 0.5
dict1 = dict()
sum =0
d=100
n = 100
theta = 0.5
print("#(head, tails) for", d, "flips and", n ,"tries with theta=" ,theta)
for times in range (0,n):
        z = generate_flip(theta,d)
        if z not in dict1:
            dict1[z] = 1
        else:
            dict1[z] =dict1[z]+1
print("The probability of (Head, Tail)>>>")
for key in dict1:
    print(key, dict1[key]/(n))
    sum += (dict1[key]/(n))
print("The sum of the probabilities is:", sum, '\n')


# d = 10 and n =1000 using a simulated coin with theta = 1/3
dict1 = dict()
sum =0
d=10
n = 1000
theta = 1/3
print("#(head, tails) for", d, "flips and", n ,"tries with theta=" ,theta)
for times in range (0,n):
        z = generate_flip(theta,d)
        if z not in dict1:
            dict1[z] = 1
        else:
            dict1[z] =dict1[z]+1
print("The probability of (Head, Tail)>>>")
for key in dict1:
    print(key, dict1[key]/(n))
    sum += (dict1[key]/(n))
print("The sum of the probabilities is:", sum, '\n')

# d = 10 and n =1000 using a simulated coin with theta = 1/2
dict1 = dict()
sum =0
d=10
n = 1000
theta = 1/2
print("#(head, tails) for", d, "flips and", n ,"tries with theta=" ,theta)
for times in range (0,n):
        z = generate_flip(theta,d)
        if z not in dict1:
            dict1[z] = 1
        else:
            dict1[z] =dict1[z]+1
print("The probability of (Head, Tail)>>>")
for key in dict1:
    print(key, dict1[key]/(n))
    sum += (dict1[key]/(n))
print("The sum of the probabilities is:", sum, '\n')







