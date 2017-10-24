'''
This code is a demonstration of the variable projection method due to Golub and Pereyra.  
Please see the accompanying pdf Final_Project.pdf for a detailed description.
'''


from  neuralnetworks2 import *
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

def numerical_derivative(f,h):
    return lambda x: (f(x+h) - f(x))/h
    





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
     
    return [np.matrix(m) for m in M]


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
    


x2 = x

a = Phi_plus * y



x2 = np.arange(0., 2*np.pi, .01) 
y2 = [  (a[0] * np.cos(alpha[0]*t) + a[1] * np.cos(alpha[1] * t) )[0,0] for t in x2]

plt.clf()

plt.plot(x,y, color = 'black' )
plt.plot(x2, y2, color = 'blue')


plt.show()
