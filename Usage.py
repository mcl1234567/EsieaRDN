import NNTP
import numpy as np
import matplotlib.pyplot as plt

#X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
#C,S = np.cos(X), np.sin(X)

#plt.plot(tab, [1, 2, 3, 4, 5, 6, 7, 8, 9])

csv = np.loadtxt("./donnees/winequality-red.csv", delimiter=';', skiprows=1)

#lignes - colonnes
X = csv[:, 0:11]
Y = csv[:, 11]

lesErreurs = []

# Create a network with two input, two hidden, and one output nodes

def nbNeurones():
    # Dimension de la couche d entree
    # Nombre de neurones sur la couche cachee
    # Une note donc un seul neurone en sortie
    for i in range(50):
        # Couches de neurones
        n = NNTP.NN(11, 5*i+1, 1)

        # N : epsylon
        # Train it with some patterns
        #print "Starting bath training"
        #if i%5 == 0:
        #print "\n\n%d : \n\n" % i

        # Train is with Back Propagation Algorithm
        # Si le nb des iterations est grand, cela est plus precis
        lesErreurs.append(n.train(X, Y, iterations=500, N=0.015, M=0, Lambda=0))

    #print lesErreurs
    plt.plot(lesErreurs, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.show()

# On modifie les iterations en fonction de l'erreur
def iterations():
    # Dimension de la couche d entree
    # Nombre de neurones sur la couche cachee
    # Une note donc un seul neurone en sortie
    for i in range(1, 500):
        n = NNTP.NN(11, 50, 1)

        # N : epsylon
        # Train it with some patterns
        #print "Starting bath training"
        #if i%5 == 0:
        #print "\n\n%d : \n\n" % i

        # Train is with Back Propagation Algorithm
        # Si le nb des iterations est grand, cela est plus precis
        lesErreurs.append(n.train(X, Y, iterations=i, N=0.015, M=0, Lambda=0))

    #print lesErreurs
    plt.plot(lesErreurs, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    plt.show()

# Selon les neurones de la couche intermediaire  de 1 a 50
#nbNeurones()

# Selon les iterations  de 1 a 500
iterations()

# ali.ziat@lip6.fr

"""
0.71633018449966612,
0.50080162674234363,
0.47183298768257143,
0.46542509781646424,
0.48242632251836565,
0.48575646185397986,
0.48673881611326214,
0.49619441186942043,
0.50237896985182617,
0.49416618786915606
"""
