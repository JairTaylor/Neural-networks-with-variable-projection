# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:37:10 2016

@author: Owner
"""


K = 500 #number of training examples
numperbatch = 500
TestK = 1000 #number of test examples
#TestN1 = 5000
numbatches = 2
numstepsperbatch = 3
outlierrate = .04



layersizes = [784,50,10]







print(2+3)



a = 0



#line_offset = [0,offset+2]

BigD = [] #keep track of 28^2 data images
BigT = [] #desired output
TestD = [] #test set
TestT = [] #test set



zeroes = []
others = []
thedigits = []

mydata = open('train.csv')


for line in mydata:
    if a % 50 == 0:
        print(a)
    #if a > 10:
    #    break
    line = mydata.readline()
    #A = [float(thing)/255 for thing in line.split(',')]
    A = line.split(',')    
    
    #print("line", A,":", line) 
    pic = [float(thing)/255 for thing in A[1:]] #between 0 and 1
    
    i = int(A[0])
    v = i    
    #thedigits.append(i)
    #v = vector([0]*10)
    #v[i] = 1
    #print(i)
    if a < K:
        BigD.append(pic)
        BigT.append(i)
    if a >= K:
        TestD.append(pic)
        TestT.append(i)
#    if A[0] == 0:        
#        BigT.append(1)
#        zeroes.append(len(BigT)-1)        
#    else:
#        BigT.append(0)
#        others.append(len(BigT)-1)  
    a+=1        
    if a > K + TestK:
        break 
mydata.close() 
#print("Digits:", thedigits)

#indices = range(K+1, K + TestK)
#for i in indices:
#    TestD.append(BigD[i])
#    TestT.append(BigT[i])


#file.seek(0)

# Now, to skip to line n (with the first line being line 0), just do
#file.seek(line_offset[n])




a = 0

# #lines = 42001.




    

#plt.imshow(np.array(A[1:]).reshape((28, 28)), cmap = cm.Greys_r)
#plt.show()






#D = [(0,0),(0,1),(1,0),(1,1)]
#T = [0,1,1,0]


tolerance = .00001

#N = len(D)


##########################
#Start new network!!!
Net = NeuralNetwork(layersizes, Ltype = "squares")
Net.RandomizeWeights(-2)
##########################

#Net.RandomizeWeights()
#Net.RandomizeWeights()


besterror = 10000


#for blah in range(10):  #first randomly generate weights and take lowest error.
#    Net.RandomizeWeights()
#    Outputs = []
#    for k in range(N):
#        Net.Propogate(vector(D[k]))
#        Outputs.append(Net.Output)
#    Error = logit(Outputs, T)
#    print("trial", blah, ": Error = ", Error)
#    if Error < besterror:
#        BestWeights = [Net.WeightMatrix, Net.Thresholds]
#        besterror = Error
#        print("better!")
#    else:
#         print("")
#Net.WeightMatrix = BestWeights[0]
#Net.Thresholds = BestWeights[1]    


#eta = 1

train = True

beta = 2
Error = 0

numback = 0

#layer = 1
#layers = [1,2]0

steplog = int(1)
batchnum = 0

while batchnum < numbatches and train:
    if batchnum == 0:
        layers = None
    else:
        layers = [1,2]
    #pic1 = Net.WeightMatrix[0][0,:].reshape(28,28)
    #pic2 = Net.WeightMatrix[0][5,:].reshape(28,28)
    #pic3 = Net.WeightMatrix[0][10,:].reshape(28,28)
    #if Error > 3:
    #    break
    print("Batch ", batchnum,":")
    T = []
    D = []
    Errors = []
    
    #indiceszero = [zeroes[i] for i in np.random.choice(K1, N1)]
    #indiceszero.sort()
    #indiceselse = [others[i] for i in np.random.choice(K2, N2)]
    #indiceselse.sort()
    
    #indices = indiceszero + indiceselse

    ########## Fix later!!!!!

    
    indices = np.random.choice(K,numperbatch, replace = False)  
    #indices.sort()

    for i in indices:
        D.append(BigD[i])
        T.append(BigT[i])
    Net.LoadInputsAndDesiredOutputs(D,T)   


    stepnum = 0
    while stepnum < numstepsperbatch:
        #if Error >3:
        #    break
#        Net.ResetDer()
#        Outputs = []
#        for k in range(len(D)):
#            Net.Propogate(vector(D[k]))
#            Outputs.append(Net.Output)
#            Net.BackPropogate(T[k])
#        Error = meansquared(Outputs,T)[0,0]
        Error = Net.ErrorWProp()
        Errors.append(Error)

        #if layer is None:
        b = max([ abs(Net.DWeightMatrix[i]).max() for i in range(len(layersizes)-1)])
        
        #
        #,abs(Net.DThresholds[-1]).max())
        #else:
            #b = max(abs(Net.DWeightMatrix[layer-1]).max(),abs(Net.DThresholds[layer-1]).max())
        #    b = (Net.DWeightMatrix[layer-1]* Net.DWeightMatrix[layer-1].transpose()).trace()[0,0]
        #    b += (Net.DThresholds[layer-1]* Net.DThresholds[layer-1].transpose()).trace()[0,0]
        #c = b * beta**steplog
        #if Error > 1.0:            
        #    print("whoooa there, Error = ", Error, "maxweight = ",b)
        #    if b > 1:              
        #        Net.Slow = b
            #else: 
            #    Net.Slow = 10
        
            #steplog -= max(0,np.ceil(np.log2(b))) +1
                #eta = eta/(2*b)
                                
                #Net.StepGradient(1/b,1)
                
                #continue
        
        
        if not Error > 0:
            break
        if Error == 10000:
            break            
        if Error < tolerance:
            break
        if not Error > 0:
            break  
        #if Error == 1.0:
        #    raise ValueError


        if len(Errors) > 1 and Errors[-1] > Errors[-2]:
            print("Error increased to ", Error, "! Backpedaling...")
            
            #Net.StepGradient(-eta)
            Net.Restore()

            if Net.Error() > Errors[-1]:
                print('wtf, not saved...')
                raise ValueError
            steplog -= 1    
            #eta = eta * beta
            Net.StepGradient(beta**steplog,layers)
            Errors = Errors[:-1]
            numback += 1
            #if numback > 10:
                
            #    np.save('savedweights', [Net.layersizes, Net.OldWeightMatrix, Net.OldThresholds])
            #    raise ValueError("Too many backpedals!!")
            continue
        else:            
                        
            print("Error", stepnum, ":", Errors[-1], "steplog:", steplog)
            stepnum += 1
            Net.SaveWeights()
            SavedInfo = [Net.WeightMatrix, Net.Thresholds, Errors[-1]]
            #np.save('savedweights', [Net.WeightMatrix, Net.Thresholds, Errors[-1]])
        numback = 0
        sig = 1/2
            
        
        if np.random.random() < .2:
            steplog += 1
            print ("trying increasing step size...")

        Net.StepGradient(beta**steplog,layers)       
        
     
    print("Before regression, error:", Net.Error())                        
    Net.LastLayerRegression()
    print("After regression, error:", Net.Error())   
     
     
     

    Outputs = []

    for k in range(len(D)):
        Net.Propogate(vector(D[k]))
        Outputs.append(Net.Output)
        #Outputs.append(Net.Output)
    # [np.round(Outputs[i]) == T[i] for i in range(len(D))].count(True) / float(len(D)) * 100
    
    p = [max(range(10), key = lambda j: Outputs[i][j]) == T[i] for i in range(len(T))].count(True) / float(len(T)) * 100
    
    
    print("We have...", p, "% correct on training set.")







    Outputs = []

    for k in range(len(TestD)):
        Net.Propogate(vector(TestD[k]))
        Outputs.append(Net.Output)
        #Outputs.append(Net.Output)
    # [np.round(Outputs[i]) == T[i] for i in range(len(D))].count(True) / float(len(D)) * 100
    
    p = [max(range(10), key = lambda j: Outputs[i][j]) == TestT[i] for i in range(len(TestT))].count(True) / float(len(TestT)) * 100
    
    
    print("We have...", p, "% correct on test set.")
    
    batchnum += 1    
    
    #Net.AddNode(1)    
    
    #savedweights.append(    [Net.layersizes, Net.WeightMatrix, Net.Thresholds, p])        
#print(Outputs)

#np.save('savedweights', savedweights)



#Net1.Load(BigD,BigT)
#Net1.LastLayerRegression()
#print("Error after regression:", Net1.Error())


#Outputs = []
#
#for k in range(len(TestD)):
#    Net1.Propogate(vector(TestD[k]))
#    Outputs.append(Net1.Output)
#    #Outputs.append(Net.Output)
## [np.round(Outputs[i]) == T[i] for i in range(len(D))].count(True) / float(len(D)) * 100
#
#
#
#p = [max(range(10), key = lambda j: Outputs[i][j]) == TestT[i] for i in range(len(TestT))].count(True) / float(len(TestT)) * 100
#



#print("We have...", p, "% correct on test set.")
