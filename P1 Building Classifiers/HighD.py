from math import *
import random
from numpy import *
import matplotlib.pyplot as plt
import numpy

waitForEnter=False

def generateUniformExample(numDim):
    return [random.random() for d in range(numDim)]

def generateUniformDataset(numDim, numEx):
    return [generateUniformExample(numDim) for n in range(numEx)]

def computeExampleDistance(x1, x2):
    dist = 0.0
    for d in range(len(x1)):
        dist += (x1[d] - x2[d]) * (x1[d] - x2[d])
    return sqrt(dist)

def computeDistances(data):
    N = len(data)
    D = len(data[0])
    dist = []
    for n in range(N):
        for m in range(n):
            dist.append( computeExampleDistance(data[n],data[m])  / sqrt(D))
    return dist

def exampleDistance(x1, x2):
    dist = 0.0
    for key_x1,val_x1 in x1.items():
        val_x2 = 0.0
        if key_x1 in x2:
            val_x2 = x2[key_x1]
        dist = dist + math.pow((val_x1 - val_x2),2)
    for key_x2,val_x2 in x2.items():
        if not key_x2 in x1:
            dist = dist + math.pow(val_x2,2)
    return sqrt(dist)

def subsampleExampleDistance(x1,x2,D):
    p = numpy.random.permutation(784)
    x1prime,x2prime = {},{}
    for n in range(D):
        Key = p[n]
        if Key in x1 and Key in x2:
            x1prime[n+1]  = x1[Key]
            x2prime[n+1] = x2[Key]
        
    return exampleDistance(x1prime, x2prime)/sqrt(D)

N    = 200                   # number of examples
Dims = [2, 8, 32, 128, 512]   # dimensionalities to try
Cols = ['#FF0000', '#880000', '#000000', '#000088', '#0000FF']
Bins = arange(0, 1, 0.02)

plt.xlabel('distance / sqrt(dimensionality)')
plt.ylabel('# of pairs of points at that distance')
plt.title('dimensionality versus uniform point distances')

for i,d in enumerate(Dims):
    distances = computeDistances(generateUniformDataset(d, N))
    print("D={0}, average distance={1}".format(d, mean(distances) * sqrt(d)))
    plt.hist(distances,
             Bins,
             histtype='step',
             color=Cols[i])
    if waitForEnter:
        plt.legend(['%d dims' % d for d in Dims])
        plt.show(False)
        x = raw_input('Press enter to continue...')


plt.legend(['%d dims' % d for d in Dims])
plt.savefig('fig.pdf')
plt.show()

