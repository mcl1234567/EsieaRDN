import math
import numpy as np

"""

rendez-vous sur
http://webia.lip6.fr/~mapsi/pmwiki.php?n=Main.TutoPython

suivez le tutoriel de la partie 1 et ecrivez le code qui repond a l exercice de synthese


"""

"""
ouvrez le fichier first_step.py
il faut maitriser tout ce qui y est decrit
Une fois fait, completez les fonctions suivantes
"""


def tanh(x):
  return np.tanh(x)

def tanh_deriv(x):
  return 1.0 - np.tanh(x) ** 2

def sigm(x):
  return 1/(1 + np.exp(-x))

def sigm_derivative(x):
  return sigm(x) * (1 - sigm(x))

#X a 4 colonnes, ci est la ieme colonne,
# Renvoie 3*c1 + 2*c2 - 5*c3 + 1*c4
def polynom(X):
  return(3 * X[:,0] + 1 * X[:,1] + 5 * X[:,2] - 3 * X[:,3])
  #return(3 * X[0,:] + 1 * X[1,:] + 5 * X[2,:] - 3 * X[3,:])

# ROOT MEAN SQUARE ERROR
def rmse(predicted, observed):
  mse = ((predicted - observed) ** 2).mean()
  return math.sqrt(mse)

def linearRegression(X, Y, epsilon, nbIteration):
  V = np.random.rand(X.shape[1])
  for i in range(nbIteration):
    pred = np.dot(X, V)
    cost = sum((pred-Y) ** 2)
    print("erreur : " + str(cost))
    V = V + epsilon * np.dot(X.transpose(), (Y-pred))
  return V

X = np.array([[1, 2, 3, 4] ,[10, 8, 0, 4], [1, 1, 1, 5], [41, 8, 3.7, 14]])
Y = np.array([14.8, 24, 13, 74])

#A = np.ones((5, 5))
#print(polynom(A))

linearRegression(X, Y, 0.0000001, 500)
