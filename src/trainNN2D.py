import pickle

#training parameters for neural networks:
learning_rate = 0.01 #set in [0,1]

learning_decay = 0.999 #try 0.999, set in [0.9,1]

momentum = 0.3 # set in [0,0.5]

batch_learning = False #set to learn in batches

validation_proportion = 0. # set in [0,0.5]
maxEpochs = 20  # max epochs to use when running to convergance

hidden_layers = [20,80,20] #number of neurons in each hidden layer, make as many layers as you feel like. Try increasing this to 10

iterations = 100 #used only if validation proportion is 0

#include/import the libraries we need for loading CSV files -------------------------------------------------------------------------------------------------------------------
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities import one_to_n

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
            # out_data.append ()
            # if out_data == 'RED': out_data.append('0')
            # else : out_data.append('1')
            out_data.append([float(i) for i in samples[-outputs:]])
        
    
    processed_out_data = out_data
    
    #process multiclass encoding
    if multiclass:
        processed_out_data = []
        #get all the unique values for classes
        keys = []
        for d in out_data:
            if d not in keys:
                keys.append(d)
        keys.sort()
        #encode all data => index pointing to each key / class encoding
        for d in out_data:
            processed_out_data.append(one_to_n(keys.index(d),len(keys)))
    
    #create the dataset
    dataset = SupervisedDataSet(len(in_data[0]),len(processed_out_data[0]))
    for i in xrange(len(out_data)):
        dataset.addSample(in_data[i],processed_out_data[i])
    
    #return the keys if we have
    if multiclass:
        return dataset,keys # a multiclass classifier
    else:
        return dataset


#train the neural network ---------------------------------------------------------------------------------------------------------------------------------------------------
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.supervised.trainers.evolino import EvolinoTrainer
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.structure import TanhLayer
from pybrain.structure.modules.evolinonetwork import EvolinoNetwork

datafilename = "synthetic_control.data.csv"
data,keys = loadCSV(datafilename)
nn = buildNetwork(*([data.indim]+hidden_layers+[data.outdim])
# #hiddenclass=TanhLayer \
)

#nn = EvolinoNetwork(data.outdim)
#trainer = BackpropTrainer(nn,data)
trainer = BackpropTrainer(nn,data,learningrate=learning_rate,momentum=momentum)
#trainer = BackpropTrainer(nn,data,learningrate=learning_rate,momentum=momentum,lrdecay=learning_decay,batchlearning=batch_learning)
#trainer = EvolinoTrainer(
 #   nn,
  #  dataset=data,
   # subPopulationSize = 20,
    #nParents = 8,
    #nCombinations = 1,
    #initialWeightRange = ( -0.01 , 0.01 ),
#    initialWeightRange = ( -0.1 , 0.1 ),
#    initialWeightRange = ( -0.5 , -0.2 ),
    #backprojectionFactor = 0.001,
    #mutationAlpha = 0.001,
#    mutationAlpha = 0.0000001,
  #  nBurstMutationEpochs = numpy.Infinity,
 #   wtRatio = wtRatio,
    #verbosity = 2)
#trainer = RPropMinusTrainer()
trainer_name = "BackProp"

error = []
validation_error = []


trainToConv = 0

# if training to converge use max epochs for iterations label
if validation_proportion>0.: iterations=maxEpochs; trainToConv =1;

filename = "{0}.{1}_epochs{2}_2x{3}x1_LR{4}_LD{5}_mom{6}.pk1".format(datafilename, trainer_name,iterations,hidden_layers,learning_rate,learning_decay,momentum)

print "Training.."
print filename

if validation_proportion == 0. or validation_proportion == 0:
    for i in xrange(iterations):
        print "up to iteration ",i
        error.append(trainer.train())

else:
    print "until convergence "
    error,validation_error = \
        trainer.trainUntilConvergence(validationProportion=validation_proportion, maxEpochs=5000, continueEpochs=25)



#save the neural network
#pickle.dump(nn, open('NN.pkl','wb'))
pickle.dump(nn, open(filename,'wb'))




###### code to save activation plot results and error plot ############


import pickle
from numpy import arange,meshgrid,zeros
import matplotlib.pyplot as plt

X = arange(-6.,6.,0.2)
Y = arange(-6.,6.,0.2)
X,Y = meshgrid(X,Y)
Z = zeros(X.shape)

picklejar = []
#filename = 'spirals.csv.BackProp_epochs1000_2x[60]x1_LR0.01_LD0.999_mom0.8.pk1'
picklejar.append(filename)

for p in range(len(picklejar)):

    model = pickle.load(open(picklejar[p]))


    for i in range(len(X)):
        for j in range(len(Y)):
            result = model.activate([X[i][j],Y[i][j]])
            if result[0] > result[1]:
                Z[i][j] = 0 #lower limit
            else:
                Z[i][j] = 100 #higher limit




    plt.imshow(Z)
    plt.gcf()
    plt.clim()

    plt.title("{0}".format(filename))
    plt.savefig('{0}.png'.format(filename), bbox_inches='tight')
    #plt.show()


### plot error
#visualise training error --------------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plot

#set up a figure
plot.figure(1)

#select the first of 2 subplots
plot.subplot(211)

#graph the error
err, = plot.plot(error,color='r')

#set the legend for the graph
legend = [[err],["Error"]]

#if there's a validation error, graph it
if len(validation_error):
    v_err, = plot.plot(validation_error,color='b')

    #add validation error to the graph
    legend[1]+=["Validation Error"]
    legend[0]+=[v_err]

#set the X and Y axis labels, and create the legend
plot.ylabel('Error')
plot.xlabel('Training Iterations')
plot.legend(*legend)

#visualise final classifications --------------------------------------------------------------------------------------------------------------------------------------------
classes = keys #range(len(keys))
width = 0.5

#select the next sub-plot
plot.subplot(212)

#get results for the whole data set:
raw_output = nn.activateOnDataset(data).tolist()
# print raw_output

#for each result, compare it to the expected result
errors = [0]*len(keys)
correct = [0]*len(keys)
for i in xrange(len(raw_output)):

    #compare the max activation values to check the classification
    d = data.getSample(i)[1].tolist()
    index = raw_output[i].index(max(raw_output[i]))


    #check if correct
    if index == d.index(max(d)):
        correct[index] += 1
    else:
        errors[index] += 1

#do the bar graphs
corr = plot.bar([i for i in xrange(len(keys))],correct,width,color='b')
err = plot.bar([i for i in xrange(len(keys))],errors,width,bottom=correct,color='r')

#label the x and y axes
plot.ylabel("# Classifications")
plot.xticks([i+width/2 for i in xrange(len(keys))],keys)

#do the legend
#plot.legend([corr,err],["Correct","Error"])

#show the plot
#plot.show()
# save the plot
plot.savefig('{0}_error.png'.format(filename), bbox_inches='tight')

from pybrain.datasets import dataset


### tried to get validator to work but couldnt ###

# from pybrain.tools.validation import CrossValidator
#
# validation_data, validation_keys = loadCSV('spirals_validation.set.csv')
# cv = CrossValidator(trainer,validation_data)
# print(cv.validate())

