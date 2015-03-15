"""
Neural Nets
Back Propagation

# Ali Ziat
"""
import numpy as np
import cPickle
import time


def tanh(x):
  return np.tanh(x)


def tanh_deriv(x):
  return 1.0 - np.tanh(x)**2

def logistic(x):
  return 1/(1 + np.exp(-x))

def logistic_derivative(x):
  return logistic(x)*(1-logistic(x))

class NN:


    def __init__(self, ni, nh, no, activation='tanh'):

        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        # nombre de neurones en input, hidden, output
        self.ni = ni + 1 # +1 pour le biais
        self.nh = nh
        self.no = no

        # couche activation
        self.ai = np.ones(self.ni)
        self.ah = np.ones(self.nh)
        self.ao = np.ones(self.no)

        # poids
        self.wi = np.random.uniform(-1.0,1.0,(self.ni, self.nh))
        self.wo = np.random.uniform(-1.0,1.0,(self.nh, self.no))


        # dernier changement pour un momentum
        self.ci = np.zeros((self.ni, self.nh))
        self.co = np.zeros((self.nh, self.no))

    #sauvegarder les poids
    def SaveW(self,filename):
         W = [self.wi,self.wo]
         cPickle.dump(W,open(filename,'w'))

    #charger les poids
    def LoadW(self,filename):
         W = cPickle.load(open(filename,'r'))
         self.wi=W[0]
         self.wo=W[1]


    #forward
    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError, 'pas le bon nombre de neurone en input'

        # input activations
        self.ai[0:self.ni-1]=inputs


        # hidden activations
        sumh = np.dot(np.transpose(self.wi),self.ai)
        self.ah = self.activation(sumh)

        # output activations
        sumo = np.dot(np.transpose(self.wo),self.ah)
        #self.ao = self.activation(sumo)   #si tanh est en activation en output
        self.ao = sumo


        return self.ao

    """Backprop
       N=pas de gradient
       M=momentum
    """


    def update_dataset(self,X):
        X=np.hstack((X,np.ones((X.shape[0],1))))
        Y=np.dot(tanh(np.dot(X,self.wi)),self.wo)
        return Y

    def backPropagate(self, targets, N, M,Lambda):

        #if len(targets) != self.no:
        #    raise ValueError, 'pas le bon nombre de neurone en output'

        # error terms for output
        #output_deltas =  self.activation_deriv(self.ao) * (targets-self.ao) #si tanh est en activation en output
        output_deltas =  (targets-self.ao)


        # calculate error terms for hidden
        error = np.dot(self.wo,output_deltas)
        hidden_deltas =  self.activation_deriv(self.ah) * error

        # update output weights
        change = output_deltas * np.reshape(self.ah,(self.ah.shape[0],1))
        self.wo = self.wo + N  * change + M * self.co
        self.wo = self.wo - Lambda*self.wo
        self.co = change


        # update input weights
        change = hidden_deltas * np.reshape(self.ai,(self.ai.shape[0],1)) - Lambda*self.wi
        self.wi = self.wi + N*change + M*self.ci
        self.wi = self.wi - Lambda*self.wi
        self.ci = change


        # calculate error
        error = sum((targets-self.ao)**2)
        error+=Lambda* (sum(sum(self.wi ** 2)) +sum(sum(self.wo ** 2)))
        return error/self.no


    """ TRAIN BATCH"""
    def train(self, X,Y, iterations=100, N=0.5, M=0,Lambda=0.002):
        # N: learning rate
        # M: momentum factor
        for i in xrange(iterations):
            error = 0.0

            for j in range(X.shape[0]):
              inputs=X[j,]
              self.update(inputs)
              error = error + self.backPropagate(Y[j,], N, M,Lambda)
            if i % 1 == 0 and i!=0:
              print 'error ' + str(error/X.shape[0])
        return error/X.shape[0]

    """ TRAIN Stochastique"""
    def stochastic_train(self, X,Y, iterations=1000, N=0.5, M=0,Lambda=0.002):
        # N: learning rate
        # M: momentum factor
        for i in xrange(iterations):
            error = 0.0
            j=np.random.randint(X.shape[0])
            inputs=X[j,]
            self.update(inputs)
            error = error + self.backPropagate(Y[j,], N, M,Lambda)
            if i % 5 == 0 and i!=0:
                print 'error ' + str(error) + 'iteration ' + str(i)

    def train_one_example(self, X,Y, N=0.5, M=0,Lambda=0.002):
        self.update(X)
        error = self.backPropagate(Y, N, M,Lambda)

def demo():
      # Teach network XOR function
      X=np.array([[0,0],[0,1],[1,0],[1,1]])
      Y= np.array([0,1,1,2])

      # create a network with two input, two hidden, and one output nodes
      a = time.clock()
      n = NN(2, 3, 1)
      #train it with some patterns
      print "Starting bath training"
      n.train(X,Y,iterations=500,N=0.015,M=0.05,Lambda=0.00002)  # Train is with Back Propagation Algorithm

      b=time.clock()
      print "Total time for Back Propagation Trainning ",b-a
      for i in range(X.shape[0]):
        res=n.update(X[i,])
        print res



def demo2():
      X=np.random.rand(3,5)

      # create a network with two input, two hidden, and one output nodes
      n = NN(5, 5, 2)
      a = time.clock()
      Y=n.update_dataset(X)
      b = time.clock()
      print(Y)
      print("time elapsed methode 1",b-a)
      a = time.clock()
      Y=n.update_dataset2(X)
      b = time.clock()
      print(Y)
      print("time elapsed avec methode 2",b-a)
