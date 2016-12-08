#training parameters for support vector machines:
from svm import *
from svmutil import svm_save_model
import pickle
from numpy import arange,meshgrid,zeros
import matplotlib.pyplot as plt

#The C parameter defines how high the cost is of miss-classification versus
# making the decision plane more complex.
# A low C will attempt to make the decisions very simple,
# but often creates classification errors in the process.
c = 50.

# gamma is a generalisation parameter,
# with low gamma corresponding to higher generalisation (and perhaps error)
# and high gamma corresponding to low generalisation (and perhaps more correct in the original dataset)
gamma = 150.


kernel = RBF

#include/import the libraries we need for loading CSV files -------------------------------------------------------------------------------------------------------------------


def loadCSV(filename,multiclass=True,outputs=1,separator=','):
    #read in all the lines
    f = open(filename).readlines()
    
    #start our datasets
    in_data = []
    out_data =[]
    
    #process the file
    for line in f:
        #remove whitespace and split according to separator character
        samples = line.strip(' \r\n').split(separator)
        
        #save input data
        in_data.append([float(i) for i in samples[:-outputs]])
        
        #save output data
        if multiclass:
            out_data.append(samples[-1])
        else:
            out_data.append([float(i) for i in samples[-outputs:]])
        
    #process multiclass encoding
    keys = []
    if multiclass:
        processed_out_data = []
        #get all the unique values for classes
        keys = []
        for d in out_data:
            if d not in keys:
                keys.append(d)
        keys.sort()
    
    #use libsvm's data container:
    return svm_problem([keys.index(i) for i in out_data],in_data),in_data,[keys.index(i) for i in out_data],keys
    


#train the SVM ---------------------------------------------------------------------------------------------------------------------------------------------------
#include the current path for library importing so that we can find the svm files
import sys
import os
path = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(path)

#set up more svm library stuff
from svmutil import svm_train,svm_predict, svm_model, svm_problem
svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

#load data
datafilename = "synthetic_control.data.csv"
dataset,data,outputs,keys = loadCSV(datafilename)




#set parameters
parameters = svm_parameter()
parameters.kernel_type = kernel
parameters.C = c
parameters.gamma = gamma
#svm_parameter.coef0
filename = "{0}_gamma{1}_c{2}.pk1".format(datafilename,gamma,c)

print "Training.."
print filename

#train
solver = svm_train(dataset,parameters)




#Save trained SVM 
#--------------------------------------------------------------------------------------------------------------------------------------------
svm_save_model(filename,solver)


