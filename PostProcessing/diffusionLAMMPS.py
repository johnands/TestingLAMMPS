import numpy as np
import matplotlib.pyplot as plt

numberOfFiles = 11

files = []
for i in range(numberOfFiles):
	files.append(open('../Argon/Data/diffusion%d.dat' % (i*100), 'r'))

# skip lines
for i in range(numberOfFiles):
	for _ in range(3):
		files[i].readline()

# number of atoms
N = int(files[0].readline())
print 'Number of atoms: %d' % N

# skip lines
for i in range(numberOfFiles):
	for _ in range(6):
		files[i].readline().strip()

# read data
displacement = np.zeros(numberOfFiles)
for i in range(numberOfFiles):
	for line in files[i]:
		if line:
			displacement[i] += float(line)**2

# normalize
displacement /= float(N)

time = range(0, 1001, 100)
plt.plot(time, displacement)
plt.show()

