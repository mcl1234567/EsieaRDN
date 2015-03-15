import NNTP
import numpy as np

csv = np.load("/home/sb/Téléchargements/rdn/winequality-red.csv")

X = csv[0:,]
Y = csv[,:11]

# Create a network with two input, two hidden, and one output nodes

# Nombre de neurones : Dimension de la couche d entree
# Nombre de neurones sur la couche cachee
# Une note, donc un seul neurone en sortie
n = NNTP.NN(11, 50, 1)

#fichier = open("/home/sb/Téléchargements/rdn/winequality-red.csv", "r")

# N : epsylon
# Train it with some patterns
print "Starting bath training"
n.train(X, Y, iterations=500, N=0.015, M=0, Lambda=0)  # Train is with Back Propagation Algorithm

fichier.close()
