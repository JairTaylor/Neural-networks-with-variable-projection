"""
Created on Wed Dec 23 12:47:11 2015

@author: Owner
"""

import numpy   as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from copy import deepcopy
from sklearn import linear_model, datasets




class NeuralNetwork:
    def __init__(self, layersizes, Ltype = "squares"):        
        self.layersizes = layersizes  
        #layersizes is list of layers; e.g., [10,40,10] means 10 input nodes, 40 hidden nodes, 10 output nodes.
        
        
        
        self.numlayers = len(layersizes)
        self.WeightMatrix = [zeromatrix(layersizes[i+1], layersizes[i]) for i in range(len(layersizes)-1)]
        self.Thresholds = [vector([0.0]*l) for l in layersizes]
        #WeightMatrix and Thresholds are lists of matrices and vectors, respectively, that give the mapping between layers;
        # that is, in propogation layer[i+1] = pointwise sigmoid( WeightMatrix[i] * layer[i] + Threshold[i] )
        
        self.DWeightMatrix = [zeromatrix(layersizes[i+1], layersizes[i]) for i in range(len(layersizes)-1)]
        self.DThresholds = [vector([0.0]*l) for l in layersizes]        
        
        #DWeightMatrix is the list of matrices of the partial derivatives dE/dw for w one of the weights between nodes,
        # where E is the error on the training set. It is set to 0 initially but found with backpropogation.
        #Similarly DThresholds is the list of partial derivatives of the thresholds.
        
        
        
        self.Ltype = Ltype
        #Ltype is the type of error used - currently logit loss "logit" and L_2-loss "squares" are implemented.
        #logit loss might be buggy.
        
        self.Output = 0
        #Output is the last layer after propogation.        
        
        self.OldWeightMatrix = self.WeightMatrix
        self.OldThresholds = self.Thresholds
        
        #OldWeightMatrix and OldThresholds are saved copies of WeightMatrix and Treshold so that they can 
        #be restored if they out perform the current weights       
        
        self.D = [0]*layersizes[0]
        self.T = [0]*layersizes[-1]  
        #D is a list of all the vectors to be inputted.  T is a list of the desired outputs.
        
              
        self.Slow = 1

        
              
    def LastLayerRegression(self):
        #Use logistic regression, imported from scikit-learn,  to set weights on last layer.
        #return True
        
        LastHiddenLayerValues = []
        Outputs = []
        for k in range(len(self.D)):
            self.Propogate(vector(self.D[k]))
            LastHiddenLayerValues.append(unvector(self.O[-2]))
        logreg = linear_model.LogisticRegression(C=1e5, solver = 'newton-cg')
        logreg.fit(LastHiddenLayerValues, self.classlabel)    
        self.WeightMatrix[-1] = logreg.coef_
        self.Thresholds[-1] = vector(logreg.intercept_)
        
    def Propogate(self, input):
        self.O = [input]    
        for i in range(self.numlayers-1):
            #print("layer",i)
            self.O.append(  sigmavec( (self.WeightMatrix[i] * self.O[i]   + self.Thresholds[i+1] )   ))  #/ float(layersizes[i])
        self.Output = self.O[-1]  #assumes output is one-dim           
        #return self.O[-1]
    def ResetDer(self):
        self.DWeightMatrix = [zeromatrix(layersizes[i+1], layersizes[i]) for i in range(len(layersizes)-1)]
        self.DThresholds= [vector([0.0]*l) for l in layersizes]        
        self.O = None
    def ResetWeights(self):
        self.WeightMatrix = [zeromatrix(layersizes[i+1], layersizes[i]) for i in range(len(layersizes)-1)]
        self.Thresholds= [vector([0.0]*l) for l in layersizes]   

    def BackPropogate(self, DesiredOutput):
        #print('yo!')
        self.DerLayer = [0] * self.numlayers
        if self.Ltype == "squares":
            self.DerLayer[self.numlayers - 1] =  self.O[-1] - DesiredOutput
        elif self.Ltype == "logit":
            if DesiredOutput == 1:
                self.DerLayer[self.numlayers - 1] =  vector([-1/self.O[-1][0,0]])
            elif DesiredOutput == 0:
                self.DerLayer[self.numlayers - 1] =  vector([1/(1-self.O[-1][0,0])])
            else:
                raise ValueError("error: Desired output must be 0 or 1")
        for l in range(self.numlayers-2, -1, -1):
            self.DerLayer[l] = self.WeightMatrix[l].transpose() * vector([self.DerLayer[l+1][i,0] * self.O[l+1][i,0] * (1 - self.O[l+1][i,0]) for i in range(len(self.O[l+1])) ] )
            #print self.DerLayer
            self.DWeightMatrix[l] += vector([self.DerLayer[l+1][i,0] * self.O[l+1][i,0] * (1 - self.O[l+1][i,0]) for i in range(len(self.O[l+1])) ]) * self.O[l].transpose()
            self.DThresholds[l+1] += vector([self.DerLayer[l+1][i,0] * self.O[l+1][i,0] * (1 - self.O[l+1][i,0]) for i in range(len(self.O[l+1])) ])
            #print "well"
            #print self.DWeightMatrix[l]
            #print self.WeightMatrix[l]
        
    def StepGradient(self,stepa,layers = None):
        if self.Slow != 1:
            step = stepa/self.Slow
        else:
            step = stepa
        if layers is None:
            for i in range(len(self.WeightMatrix)):
                self.WeightMatrix[i] -= step*self.DWeightMatrix[i]
            for i in range(len(self.Thresholds)):
                self.Thresholds[i] -= step*self.DThresholds[i]
        else:
            for i in layers:
                self.WeightMatrix[i-1] -= step*self.DWeightMatrix[i-1]
            for i in layers:
                self.Thresholds[i-1] -= step*self.DThresholds[i-1]
        self.Slow = 1    

    def LoadWeightsFromFile(self,filename):
        #load weights saved to disk
        (self.WeightMatrix, self.Thresholds, p) = np.load(filename)

    def SaveWeights(self):
        #save current we
        self.OldWeightMatrix = deepcopy(self.WeightMatrix)
        self.OldThresholds = deepcopy(self.Thresholds)
        #print("                         saving!", (self.WeightMatrix[0]).max(),(self.WeightMatrix[1]).max())
    def Restore(self):
        self.WeightMatrix = deepcopy(self.OldWeightMatrix)
        self.Thresholds = deepcopy(self.OldThresholds)
        
    def RandomizeWeights(self,mu):
        self.WeightMatrix = [random_matrix(self.layersizes[i+1], self.layersizes[i]) for i in range(self.numlayers-1)]
        self.Thresholds= [ random_matrix(l,1,mu)  for l in self.layersizes]
    #def RandomizeDWeights(self):
    #    self.DWeightMatrix = [random_matrix(self.layersizes[i+1], self.layersizes[i]) for i in range(self.numlayers-1)]
    #    self.DThresholds= [ random_matrix(l,1) for l in self.layersizes]    
        
    def LogitError(self,D,T):
        Outs = []
        for k in range(len(D)):
            self.Propogate(vector(D[k]))
            Outs.append(Net.Output)
        #print "Outs:", Outs
        return logit(Outs, T)
        
    def TestStep(self, D,T, step):
        self.StepGradient(step)
        err = self.LogitError(D,T)
        #self.StepGradient(-step)
        return err


    
    def LoadInputsAndDesiredOutputs(self,D,classlabel):
        self.classlabel = classlabel
        self.D = D
        self.T = [vectorize(i,10) for i in classlabel]
    
    def LoadInputsAndDesiredOutputsForRegression(self,D,classlabel):
        self.classlabel = classlabel
        self.D = D
        self.T = T
    
    
    def Error(self):
        Outputs = []
        for k in range(len(self.D)):
            self.Propogate(vector(self.D[k]))
            Outputs.append(self.Output)
            #self.BackPropogate(self.T[k])
        error = meansquared(Outputs,self.T)[0,0] 
        #if error == 1.0:
        #    print(Outputs)
            #print(T)
        #Net    raise ValueError
        return   error
 
    def ErrorWProp(self):
        self.ResetDer()
        Outputs = []
        for k in range(len(self.D)):
            self.Propogate(vector(self.D[k]))
            Outputs.append(self.Output)
            self.BackPropogate(self.T[k])
        #Net.Outputs = Outputs
        return meansquared(Outputs,self.T)[0,0]        
        
    def CheckDWeight(self, stepsize,layer,i,j):
        #return difference quotient for checking against computed gradient.
        self.SaveWeights()
        e1 = self.ErrorWProp()
        self.WeightMatrix[layer][i,j] += stepsize
        e2 = self.Error()
        self.Restore()
        print((e2 - e1)  * self.layersizes[-1] /stepsize)
        print(self.DWeightMatrix[layer][i,j]  )
        
    def CheckDThreshold(self, stepsize,layer,i,j):
        #return difference quotient for checking against computed gradient.
        self.SaveWeights()
        e1 = self.ErrorWProp()
        self.Thresholds[layer][i,j] += stepsize
        e2 = self.Error()
        self.Restore()
        print((e2 - e1)  * self.layersizes[-1] /stepsize)
        print(self.DThresholds[layer][i,j]  )        
        


def showpic(pic): #convert a 28 x 28 vector into a picture
    plt.imshow(pic.reshape((28, 28)), cmap = cm.Greys_r)


def vectorize(i,n):
    A = [0]*n
    A[i] = 1
    return vector(A)


def zeromatrix(m,n):
    
    return np.matrix(np.zeros([m,n]))
    #return np.matrix(np.zeros([3,3]))
    
def random_matrix(m,n, mu = 0):
    #print("randomizing!!",m,n)
    return  np.matrix(mu + np.random.randn(m,n)/(m*n))

def vector(A):
    #print("WHOOA THERE",A)
    return np.matrix([A]).transpose()


def unvector(v):
    #returns list version of (column) vector.
    return [v[i,0] for i in range(len(v))]

def logit(O,T):
    #given vectors O, T, compute -log-likelihood of T given O.
    a = 0
    for i in range(len(T)):
        if T[i] == 1:
            a += np.log(O[i])
        elif T[i] == 0:
            a += np.log(1-O[i])
        else:
            raise ValueError("T must be 0-1 vector")    
    return -a

def meansquared(O,T):
    if len(O) != len(T):
        raise ValueError
    return sum([(O[i] - T[i]).transpose()*(O[i] - T[i]) for i in range(len(O))])/len(O)
    #return sum([(O[i] - T[i])*(O[i] - T[i]) for i in range(len(O))])/len(O)

def sigmavec(v):
    #print("WHOOA THERE v",v)
    w = []
    for i in range(len(v)):
        #if v[i,0] < -20 or v[i,0] > 20:
        #    print('wtf')
        #    print(v)
        #    raise(ValueError)
        #    w.append(0)
        #elif v[i,0] > 5:
        #    w.append(1)
        #else:    
        #if abs( v[i,0]) > 10:
        #    raise ValueError
        if v[i,0] > 10:
            w.append(1.0)
        elif v[i,0] < -10:
            w.append(0.0)
        else:
            w.append(  1/(1 + np.exp(-   v[i,0]  ) ))
    return vector(w)




        #print(sum([(M * M.transpose()).trace() for M in self.DWeightMatrix])[0,0] + sum([(M * M.transpose()).trace()  for M in self.DThresholds])[0,0])
#######################################

#Example of use

#import gzip
#import pickle
#with gzip.open('mnist.pkl.gz', 'rb') as f:
#    train_set, valid_set, test_set = pickle.load(f)

#K2 = K1
#N2 = N1
#TestN2 = TestN1

#N = N1 + N2
#Er

#line_offset = []
#offset = 0






#################################