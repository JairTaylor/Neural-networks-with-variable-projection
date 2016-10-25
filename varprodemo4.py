# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from  neuralnetworks2 import *
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

def numerical_derivative(f,h):
    return lambda x: (f(x+h) - f(x))/h
    


#n = 5
#



#x =  np.arange(0., 2*np.pi, 1.0)   #observed inputs


#x is a list of inputs, y a list of outputs, alpha a list of parameters.




x =  [ 0  +1.0 * i for i in range(4)   ]
y =  [3 * np.cos(4*t) + 5 * np.cos(2*t) for t in x]      #observed outputs

#x = np.matrix(x).T
y = np.matrix(y).T
n = len(x)
#



def Basis(alpha,t):
    #given parameters alpha and input t, return row of basis functions applied to x.
    Net.setweightsfromparamlist(alpha)
    Net.Propogate(t)
    return Net.O[-1].T.tolist()[0]

def JBasis(alpha,t):    
    Net.ResetDer()
    Net.setweightsfromparamlist(alpha)
    
    Net.Propogate(np.matrix(t))
    Net.BackPropogateJ()
    #Dalpha = Net.Dparamlist
    return [   [Net.Dparamlist(i)[j] for i in range(Net.layersizes[-1])] for j in range(len(alpha))]
   
   #return [ [ -t *np.sin(alpha[0] * t),0,0] ,[ 0,-t*np.sin(alpha[1] * t) ,0 ],[0,0,t* np.exp(alpha[2] * t) ] ]
    
    #given parameters alpha and input x, return Jacobian: do_i/dalpha_j


def Phi(alpha,x):
    M = [ Basis(alpha, t) for t in x]
    #M = [ [  np.cos(alpha[0] * x[i,0]),np.cos(alpha[1] * x[i,0])   ] for i in range(len(x)) ]        
    return  np.matrix(M)


def dPhi(alpha, x):
    M = []
    for i in range(len(alpha)):
        M.append([])
    for t in x:
        #print 'heyo'
        K = JBasis(alpha,t)
        for i in range(len(M)):
            M[i].append(K[i])
    
    #M0 = np.matrix(  [ [ -x[i,0] *np.sin(alpha[0] * x[i,0]),0]   for i in range(len(x)) ] ) 
    #M1 = np.matrix(  [ [ 0,-x[i,0] *np.sin(alpha[1] * x[i,0])]   for i in range(len(x)) ] )  
    return [np.matrix(m) for m in M]



#x =  np.arange(-1, 1.2, 0.1)   #observed inputs
#y =  [3 * np.exp(2*t) + 5 * np.exp(-3 * t) for t in x]      #observed outputs
#x = np.matrix(x).T
#y = np.matrix(y).T
#
#n = len(x)

#def Phi(alpha,x):
#    M = [ [  np.exp(alpha[0] * x[i,0]),np.exp(alpha[1] * x[i,0])   ] for i in range(len(x)) ]        
#    return  np.matrix(M)
#
#
#def dPhi(alpha, x):
#    M0 = np.matrix(  [ [ x[i,0]*np.exp(alpha[0] * x[i,0]),0]   for i in range(len(x)) ] ) 
#    M1 = np.matrix(  [ [ 0, x[i,0]*np.exp(alpha[1] * x[i,0])]   for i in range(len(x)) ] )
#    return (M0, M1)
#
#
def Error(alpha, x):
    Phi_plus = np.linalg.pinv(Phi(alpha,x))
    PPhi = (np.identity(n) - Phi(alpha,x) * Phi_plus    ) 
    E =  .5 * np.linalg.norm( PPhi*y)**2  
    return E
#    


#Phi = lambda t: M(t) * x
#

truealpha = [2,-3]


#alpha = [2,4.8,2]

Net = NeuralNetwork([1,5,3])
Net.RandomizeWeights(-2)
alpha = Net.paramlist()



basestep =   0

g = lambda alpha:  .5 * np.linalg.norm( (np.identity(n) - Phi(alpha,x) *   np.linalg.pinv(Phi(alpha,x))) * y) **2

#Errors = []
tol = .01

Phi_plus = np.linalg.pinv(Phi(alpha,x))
PPhi = (np.identity(n) - Phi(alpha,x) * Phi_plus    ) 
E =  .5 * np.linalg.norm( PPhi*y)**2  

Errors = [E]
print "Starting error:", E


D = dPhi(alpha,x)
J = np.hstack([ ((PPhi * d * Phi_plus )   +  (PPhi * d * Phi_plus ).T ) * -y for d in D])


#calculate gradient

grad = J.T * PPhi * y

f = lambda t: Error( [t] + alpha[1:],x)

print grad[0,0]
print numerical_derivative(f, .000000000001)(alpha[0])

numsteps = 0
oldalpha = copy(alpha)


for stepnum in range(-2):
    if Errors[-1] < tol:
        break
    #calculate Jacobian
    
 
    #print "current error:", E
    Phi_plus = np.linalg.pinv(Phi(alpha,x))
    PPhi = (np.identity(n) - Phi(alpha,x) * Phi_plus    ) 
    D = dPhi(alpha,x)
    J = np.hstack([ ((PPhi * d * Phi_plus )   +  (PPhi * d * Phi_plus ).T ) * -y for d in D])
    
    
    #calculate gradient    
    grad = J.T * PPhi * y
        
    step = basestep
    for i in range(30):  
        step = basestep - i
        alphaguess = copy(alpha)
        for i in range(len(alpha)):
            alphaguess += - np.exp(step) *  grad[i][0,0]
        #alphaguess[0] +=   - np.exp(step) *  grad[0][0,0]
        #alphaguess[1] +=    -np.exp(step) *  grad[1][0,0]
    
    
        Phi_plus = np.linalg.pinv(Phi(alphaguess,x))
        PPhi = (np.identity(n) - Phi(alphaguess,x) * Phi_plus    ) 
        E =  .5 * np.linalg.norm( PPhi*y)**2  
        if E - Errors[-1] < -.5 * grad.T * grad * np.exp(step) : #armijo condition
            alpha = copy(alphaguess)
            #basestep = basestep -i + 3
            Errors.append(E)
            print "current error:", E, "step:", step, np.round(alphaguess,4)            
            
            break
        else:
            #pass
            print "backtracking!", E, step
    else:
        print 'failed to step'
        break        
    



#x =  np.arange(0., 2*np.pi, 1.0)   #observed inputs

#a = Phi_plus*y
#
#x2 = np.arange(0., 2*np.pi, .01) 
#y2 =  [a[0,0] * np.cos(alpha[0]*t) + a[1,0] * np.cos(alpha[1]*t) for t in x2]      #observed outputs
#
#x3 = np.arange(0., 2*np.pi, .01) 
#y3 = [3 * np.cos(4*t) + 5 * np.cos(2*t) for t in x3]
#
#plt.clf()



x2 = x

a = Phi_plus * y


#y2 = [  a[0] * np.exp(alpha[0]*t) + a[1] * np.exp(alpha[1] * t) for t in x2]
#print 'y2', y2[0]
#y2 = [b[0,0] for b in y2]
#y2 = a[0] 



x2 = np.arange(0., 2*np.pi, .01) 
y2 = [  (a[0] * np.cos(alpha[0]*t) + a[1] * np.cos(alpha[1] * t) )[0,0] for t in x2]

plt.clf()

plt.plot(x,y, color = 'black' )
plt.plot(x2, y2, color = 'blue')


plt.show()




#from neuralnetworks2 import *
#
#Net = NeuralNetwork([1,5,3])
#
##Net = NeuralNetwork([1,4,2])
#Net.RandomizeWeights(-2)
#v = np.matrix(7)
#Net.ResetDer()
#Net.Propogate(v)
#
#def testd(t):
#    Net.setweightsfromparamlist([t] + alpha[1:])
#    Net.Propogate(v)
#    return Net.Output[0,0]
#    
#
#Net.BackPropogateJ()
#print Net.DThresholdsJ[0]
#    



