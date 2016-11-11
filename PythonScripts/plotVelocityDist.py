import numpy as np
import matplotlib.pyplot as plt

def extractInitial(filename):

    infile = open(filename, 'r')

    n = int(infile.readline())			# number of atoms
    
    infile.readline()				# skip comment line
	
    # initialize lists
    vx = []; vy = []; vz = []

    # extract velocities from file
    i = 0
    for line in infile:
        if i >= n:
            print i
            break
        words = line.split() 
        vx.append(float(words[4]))
        vy.append(float(words[5]))
        vz.append(float(words[6]))
        i += 1

    infile.close()

    return vx, vy, vz, n


def extractWhole(filename):

    infile = open(filename, 'r')

    noAtoms = int(infile.readline()) 	
    infile.readline()						# skip comment line
			
    # initialize lists
    velocities = []
    speeds = []

    # extract velocities from file
    
    velocities.append([])
    speeds.append([])
    timeStep = 0
    for line in infile:
        words = line.split() 
        if len(words) == 7:
            vx = float(words[4])
            vy = float(words[5])
            vz = float(words[6])
            v = np.sqrt(vx**2 + vy**2 + vz**2)
            velocities[timeStep].append([vx, vy, vz])
            speeds[timeStep].append(v)
        elif len(words) == 1:
            timeStep += 1
            velocities.append([])
            speeds.append([])
        else:
            pass

    infile.close()

    return velocities, speeds, timeStep, noAtoms




def plotInitialDistribution():
    vx, vy, vz, noAtoms = extractInitial('../Argon/argon.xyz')

    # convert to numpy arrays
    vx = np.array(vx); vy = np.array(vy); vz = np.array(vz)

    # magnitude of velocites
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # compute mean
    mean_vx = sum(vx)/n
    mean_vy = sum(vy)/n
    mean_vz = sum(vz)/n

    print "Mean vx = ", mean_vx
    print "Mean vy = ", mean_vy 
    print "Mean vz = ", mean_vz

    # compute standard deviation
    sigma_x = np.sqrt(sum((vx-mean_vx)**2)/n)
    sigma_y = np.sqrt(sum((vy-mean_vy)**2)/n)
    sigma_z = np.sqrt(sum((vz-mean_vz)**2)/n)
    print "St.dev. vx = ", sigma_x
    print "St.dev. vy = ", sigma_y
    print "St.dev. vz = ", sigma_z

    
    plt.hist(vx, bins=100)
    plt.show()

    plt.hist(v, bins=100)
    plt.show()


def animateDistribution():
    velocities, speeds, noTimeSteps, noAtoms = extractWhole('animateVelocityDist.xyz')

    # convert to numpy arrays
    velocities = np.array(velocities)
    speeds = np.array(speeds)

    print velocities.shape
    print speeds.shape
    print velocities[0][:][:,0].shape

    # animate velocities
    for t in xrange(noTimeSteps):
        if not t % 10:
            plt.hist(velocities[t][:][:,0], bins=100)
            plt.show()  
    
    """
    # animate speeds
    for t in xrange(noTimeSteps):
        if not t % 100:
            plt.hist(speeds[t][:], bins=100)
            plt.show()  
    """    
    

    

# main

plotInitialDistribution()
#animateDistribution()
