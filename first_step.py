)import numpy as np

print("hello world!")

# Creer un vecteur
vec = np.array([1, 2, 3, 4])

# Creer une matrice
mat = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])

# Afficher le vecteur et la matrice
print(mat)
print(vec)

# Multiplication d un scalaire et d une matrice
mat = 3 * mat

# Multiplication d un vecteur et d une matrice
res = np.dot(vec, mat)

# Somme des tous les elements du vecteur res
somme = sum(res)

# Creer une matrice de nombres aleatoires
nbligne = 5
nbcolonne = 3
mat = np.random.rand(nbligne, nbcolonne)

# Creer une matrice de nombres aleatoires avec des valeurs entieres
mat = np.random.randint(nbligne, nbcolonne)

# Creer une matrice de 1
m1 = np.ones((10, 2))  # matrice de 1, argument = nuplet avec les dimensions
                       # ATTENTION np.ones(10,2) ne marche pas

# Creer une matrice de 0
m0 = np.zeros((10, 2))  # matrice de 0, argument = nuplet avec les dimensions
                       # ATTENTION np.ones(10,2) ne marche pas

# Afficher la dimension de la matrice
print(mat.shape) # Resultat sous forme nbligne nbcolonne

# Selectionner une ligne dans la matrice
ligne = mat[0,] # la premiere ligne

# Selectionner une colonne dans la matrice
colonne = mat[:,2] # la troisieme colonne

# Vecteur des valeurs de 0 à 9
vec = range(10)

# Boucle sur python
for i in range(10):
  print i

# Concatener des chaines de caracteres
ch1 = "hello"
ch2 = " 42"
ch3 = ch1 + " numero " + ch2

# Longueur d un vecteur
long = len(vec)
#ou
long = vec.shape[0]

# Definir une fonction
def carre(x):
  return x ** 2

# Maximum d un vecteur
def maximum(vec)
  maxi = vec[0]
  for i in range(len(vec)):
    if vec[i] > maxi:
      maxi = vec[i]
  return maxi

#!!! attention !!!
"""
numpy dispose d une fonction max beaucoup plus rapide
maxi = np.max(vec)
de maniere generale toujours eviter les boucles en python quand c est possible

"""

# Lecture ecriture de matrice
"""
mat = np.loadtxt("fichier.txt")

np.savetxt("fichier.txt", mat)

"""
