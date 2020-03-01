import numpy as np 
import matplotlib.pyplot as plt 
import random

#data
x=np.array([63,81,56,91,47,57,76,72,62,48,65,84,59,93,49,55,79,75,66,49])
y=np.array([151,174,138,186,128,136,179,163,152,131,153,177,148,189,138,146,199,167,153,130])

#number of elements in x
n=float(len(x))

#m is slope and c is intercept(initially)
m=0
c=0

#learning rate
L=0.001

#number of iteration
epochs=1000

#stochastic gradient decent model
for i in range(epochs):
	np.random.shuffle(x) #selecting a data at random
	for example in x:
		Y=(m*x)+c
		D_m=(-2/n) * (x*(y-Y)) #gradient with respect to m
		D_c=(-2/n) * ((y-Y)) #gradient with respect to c
		m=m-(L * D_m) #updating m
		c=c-(L * D_c) #updating c

print(m)
print(c)

#best fit line
Y=(m*x)+c

plt.scatter(x,y)
plt.plot([min(x), max(x)], [min(Y), max(Y)], color='red') # predicted
plt.show()


