import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def readFile(filename):
    
    with open(filename, 'r') as infile:
        
        # skip two comment lines
        infile.readline()
        infile.readline()
        
        temperature = []
        kineticEnergy = []
        potentialEnergy = []
        pressure = []
        for line in infile:
            words = line.split()
            temperature.append(float(words[1]))
            kineticEnergy.append(float(words[2]))
            potentialEnergy.append(float(words[3]))
            pressure.append(float(words[4]))
            
    temperature = np.array(temperature)
    kineticEnergy = np.array(kineticEnergy)
    potentialEnergy = np.array(potentialEnergy)
    pressure = np.array(pressure)
    
    return temperature, kineticEnergy, potentialEnergy, pressure
    

def readPotFile(filename, chosenID):
   
    with open(filename, 'r') as infile:
        
        pot = False  
        timeStep = False                    
        energies = []
        timeSteps = []
        
        # read number of atoms and first time step
        infile.readline()
        timeSteps.append(int(infile.readline()))
        
        infile.readline()
        numberOfAtoms = int(infile.readline())
        print "Number of atoms: ", numberOfAtoms
      
        i = 0
        for line in infile:
            words = line.split()
            
            if words[-1] == 'TIMESTEP':
                timeStep = True
                continue
                
            if words[-1] == 'c_peAtom':
                pot = True
                continue
            
            if timeStep:
                timeSteps.append(int(words[0]))
                timeStep = False
                continue
                
            if pot:
                i += 1
                words = line.split()
                if int(words[0]) == chosenID:
                    energies.append(float(words[1]))
                if i == numberOfAtoms:
                    i = 0
                    pot = False 
           
    return np.array(timeSteps), np.array(energies), numberOfAtoms
    
    
    
def targetAndNN(dirNameSW, dirNameNN, plot):
    
    # write out README files
    print "Content of SW folder:"
    os.system('cat ' + dirNameSW + 'README.txt')
    print "Content of NN folder:"
    os.system('cat ' + dirNameNN + 'README.txt')
    
    tempNN, kinNN, potNN, pressNN = readFile(dirNameNN + 'thermo.txt')
    tempSW, kinSW, potSW, pressSW = readFile(dirNameSW + 'thermo.txt')
    
    if not (plot == 'notNN' or plot == 'tempFluct'):
        potFactor = np.average(potNN / potSW)
        print "Potential energy factor: ", potFactor
        #potNN /= potFactor
    
    totalEnergyNN = kinNN + potNN
    totalEnergySW = kinSW + potSW
    
    numberOfSamples = len(tempNN)
    
    print "Slope of energy drift: ", (totalEnergyNN[-1] - totalEnergyNN[0])/len(tempNN)
    
    print 
    print "Average temp NN: ", np.average(tempNN)
    print "Average temp SW: ", np.average(tempSW)
    print "Average kin NN: ", np.average(kinNN)
    print "Average kin SW: ", np.average(kinSW)
    print "Average pot NN: ", np.average(potNN)
    print "Average pot SW: ", np.average(potSW)
    print "Average tot NN: ", np.average(totalEnergyNN)
    print "Average tot SW: ", np.average(totalEnergySW)
    
    print
    print "Std. dev. temp oscillations NN: ", np.std(tempNN)
    print "Std. dev. temp oscillations SW: ", np.std(tempSW)


    if plot == 'both':
        plt.subplot(2,2,1)
        plt.plot(tempNN, 'b-', tempSW, 'g-')
        plt.subplot(2,2,2)
        plt.plot(kinNN, 'b-', kinSW, 'g-')
        plt.subplot(2,2,3)
        plt.plot(potNN, 'b-', potSW, 'g-')
        plt.subplot(2,2,4)
        plt.plot(totalEnergyNN, 'b-', totalEnergySW, 'g-')
        plt.show()
        
    elif plot == 'kin':
        plt.plot(kinNN - kinSW)
        plt.xlabel('Timestep')
        plt.ylabel('Kinetic energy')
        plt.show()
        
    elif plot == 'notNN':
        plt.subplot(2,2,1)
        plt.plot(tempSW)
        plt.subplot(2,2,2)
        plt.plot(kinSW)
        plt.subplot(2,2,3)
        plt.plot(potSW)
        plt.subplot(2,2,4)
        plt.plot(totalEnergySW)
        plt.show()
        
    elif plot == 'tempFluct':
        dev = tempSW - np.average(tempSW)
        plt.plot(dev)
        plt.show()
        
        # compute interval average oscillations
        bins = 10
        binSize = numberOfSamples/bins
        deviations = np.zeros(bins)
        for i in xrange(bins):
            deviations[i] = np.std(dev[i*binSize:(i+1)*binSize])
            
        print deviations    
        
        
    else: 
        plt.subplot(2,2,1)
        plt.plot(tempNN)
        plt.subplot(2,2,2)
        plt.plot(kinNN)
        plt.subplot(2,2,3)
        plt.plot(potNN)
        plt.subplot(2,2,4)
        plt.plot(totalEnergyNN)
        plt.show()
        
        
def multipleNNAverage(dirNames):
    
    numberOfNNs = len(dirNames)
    
    temp = []; kin = []; pot = []; tot = []
    for i in xrange(numberOfNNs):
        itemp, ikin, ipot, _ = readFile(dirNames[i] + 'thermo.txt')
        temp.append(itemp)
        kin.append(ikin)
        pot.append(ipot)
        tot.append(ikin + ipot)
        
    legends = []
    for i in xrange(numberOfNNs):
        plt.plot(pot[i])
        legends.append('NNP number %d' % i)
        
    plt.legend(legends, loc=4)
    plt.show()
    
    stdDev = np.std(pot, axis=0)
    plt.plot(stdDev)
    plt.legend(['Std. dev. of all NNPs as function of time'])
    plt.show()
    
    
def multipleNN(dirNames, chosenID, write=False):
    
    numberOfNNs = len(dirNames)
    
    pot = []
    cut = 3000
    for i in xrange(numberOfNNs):
        timeSteps, energies, _ = readPotFile(dirNames[i] + 'thermo.txt', chosenID)
        pot.append(energies)
        
    plt.rc('lines', linewidth=1.5)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=18)
        
    legends = []
    for i in xrange(numberOfNNs):
        plt.plot(pot[i][:cut])
        legends.append('NNP %d' % (i+1))
        
    plt.legend(legends, loc=2)
    plt.xlabel('Time step')
    plt.ylabel('Potential energy [eV]')
    plt.tight_layout()
    #plt.savefig('../../Oppgaven/Figures/Results/multipleNNP.pdf')
    #plt.show()
    
    stdDev = np.std(pot[:cut], axis=0)
    plt.plot(stdDev)
    plt.legend(['Std. dev. of all NNPs as function of time'])
    #plt.show()
    
    absError = np.max(pot[:cut], axis=0) - np.min(pot[:cut], axis=0)
    plt.plot(absError)
    plt.legend(['Absolute error'])
    #plt.show()
    
    avePot = np.average(pot, axis=0)
    print np.where(stdDev > np.abs(0.001*avePot))[0]
    
    samples = np.where(absError > 0.005)[0]
    print len(samples)
    
    if write:
        with open('samples10000-13000A%d.txt' % chosenID, 'w') as outfile:
            for i in xrange(len(samples)):
                outfile.write('%d' % samples[i])
                outfile.write('\n')
        
    
    
        
 
    

#dirName1 = '../TestNN/Data/Si/Thermo/MultipleNNP1/'
#dirName2 = '../TestNN/Data/Si/Thermo/MultipleNNP2/'
#dirName3 = '../TestNN/Data/Si/Thermo/MultipleNNP3/'
#dirName4 = '../TestNN/Data/Si/Thermo/MultipleNNP4/'
dirName1 = '../TestNN/Data/Si/Thermo/MultipleNNP1Restart/'
dirName2 = '../TestNN/Data/Si/Thermo/MultipleNNP2Restart/'
dirName3 = '../TestNN/Data/Si/Thermo/MultipleNNP3Restart/'
dirName4 = '../TestNN/Data/Si/Thermo/MultipleNNP4Restart/'
dirNames = [dirName1, dirName2, dirName3, dirName4]
#dirNames = [dirName1, dirName2]

multipleNN(dirNames, 200, write=True)

#targetAndNN('../Silicon/Data/Thermo/L3T300N3000/', '../TestNN/Data/Si/Thermo/L3T300N3000Pseudo/', 'both')



    


