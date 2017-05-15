"""
Plot coordination number for both pair and triplet cut SiO2
for all combinations of types
"""

import numpy as np
import matplotlib.pyplot as plt

filename = '../MultiAtom/Data/Coord/coordNumberT2000.txt'
coords = np.loadtxt(filename)

timeSteps = coords[:,0]

# min - ave - max
# pairCut - tripletCut
# Si-Si Si-O O-O O-Si

plt.subplot(2,2,1)
plt.plot(timeSteps, coords[:,1])
plt.hold('on')
plt.plot(timeSteps, coords[:,2])
plt.hold('on')
plt.plot(timeSteps, coords[:,3])
plt.legend(['Min Si-Si', 'Ave Si-Si', 'Max Si-Si'])
plt.ylabel('Coordination number')

plt.subplot(2,2,2)
plt.plot(timeSteps, coords[:,4])
plt.hold('on')
plt.plot(timeSteps, coords[:,5])
plt.hold('on')
plt.plot(timeSteps, coords[:,6])
plt.legend(['Min Si-O', 'Ave Si-O', 'Max Si-O'])

plt.subplot(2,2,3)
plt.plot(timeSteps, coords[:,10])
plt.hold('on')
plt.plot(timeSteps, coords[:,11])
plt.hold('on')
plt.plot(timeSteps, coords[:,12])
plt.legend(['Min O-Si', 'Ave O-Si', 'Max O-Si'])
plt.xlabel('Time Steps')

plt.subplot(2,2,4)
plt.plot(timeSteps, coords[:,7])
plt.hold('on')
plt.plot(timeSteps, coords[:,8])
plt.hold('on')
plt.plot(timeSteps, coords[:,9])
plt.legend(['Min O-O', 'Ave O-O', 'Max O-O'])
plt.xlabel('Time Steps')
plt.ylabel('Coordination number')

plt.show()

plt.figure()

plt.subplot(2,2,1)
plt.plot(timeSteps, coords[:,13])
plt.hold('on')
plt.plot(timeSteps, coords[:,14])
plt.hold('on')
plt.plot(timeSteps, coords[:,15])
plt.legend(['Min Si-Si', 'Ave Si-Si', 'Max Si-Si'])
plt.ylabel('Coordination number')

plt.subplot(2,2,2)
plt.plot(timeSteps, coords[:,16])
plt.hold('on')
plt.plot(timeSteps, coords[:,17])
plt.hold('on')
plt.plot(timeSteps, coords[:,18])
plt.legend(['Min Si-O', 'Ave Si-O', 'Max Si-O'])

plt.subplot(2,2,3)
plt.plot(timeSteps, coords[:,22])
plt.hold('on')
plt.plot(timeSteps, coords[:,23])
plt.hold('on')
plt.plot(timeSteps, coords[:,24])
plt.legend(['Min O-Si', 'Ave O-Si', 'Max O-Si'])
plt.xlabel('Time Steps')

plt.subplot(2,2,4)
plt.plot(timeSteps, coords[:,19])
plt.hold('on')
plt.plot(timeSteps, coords[:,20])
plt.hold('on')
plt.plot(timeSteps, coords[:,21])
plt.legend(['Min O-O', 'Ave O-O', 'Max O-O'])
plt.xlabel('Time Steps')
plt.ylabel('Coordination number')

plt.show()












        