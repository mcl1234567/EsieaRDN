import numpy as np
from random import *

# Contient une matrice de 10 elements de 1 a 11 
#var1 = np.arange(1, 11)
						#create a range
                      	#arguments: start, stop, step
 					  	#with default arg

#mat   = np.ones((10, 3))
#m0 = np.arange(1, 11)
#m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# matrice
# matrice = vecteur de vecteurs
#_rand = np.random.randn(1, 2)
#L = [0, 1]

"""m1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
				[choice(L), choice(L), choice(L), choice(L), choice(L), choice(L), choice(L), choice(L), choice(L), choice(L)], 
				[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])"""

m1 = np.zeros((10, 3))

# Premiere ligne de 1 a 3
m1[0, 1:2] = 2
m1[0, 2:3] = 3

# Premiere colonne de 1 a 10
for x in xrange(1, 10):
	m1[x, 0:1] = x

# Deuxieme colonne random 0 et 1
L = [0, 1]
for x in xrange(1, 10):
	m1[x, 1:2] = choice(L)

print(m1[::])

print("Moyenne : %s" % m1.mean())
print("Ecart-type : %s" % m1.std())

n = 15
m5 = np.random.randn(5, 6)
mat = np.arange(1, n)
print("nb aleatoire : %s" % choice(mat))
print m5

#a = np.array((1, 2, 3))
#b = np.array((4, 5, 6))
#print(np.hstack((a, b)))