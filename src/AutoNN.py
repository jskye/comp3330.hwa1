#training parameters for neural networks:
learning_rate = 0.3 #set in [0,1]

learning_decay = 0.9999 #try 0.999, set in [0.9,1]

momentum = 0.3 # set in [0,0.5]

batch_learning = False #set to learn in batches

validation_proportion = 0. # set in [0,0.5]

#hidden_layers = [5,5] #number of neurons in each hidden layer, make as many layers as you feel like. Try increasing this to 10
h1 = 1
h2 = 1
hidden_layers = [h1,h2] #number of neurons in each hidden layer, make as many layers as you feel like. Try increasing this to 10

iterations = 5 #used only if validaton proportion is 0

#include/import the libraries we need for loading CSV files -------------------------------------------------------------------------------------------------------------------
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import UnsupervisedDataSet
from pybrain.utilities import one_to_n
import numpy as np
import matplotlib.pyplot as plt

def loadCSV_auto(filename,multiclass=True,outputs=1,separator=','):
    #read in all the lines
    f = open(filename).readlines()[0:]

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
        #encode all data
        for d in out_data:
            processed_out_data.append(one_to_n(keys.index(d),len(keys)))

    #create the dataset
    data_set = SupervisedDataSet(len(in_data[0]),len(in_data[0]))
    #data_set = UnsupervisedDataSet(26)
    # data_set.addSample([0, 1] * 3)
    # data_set.addSample([1, 0] * 3)

    for i in xrange(len(out_data)):
        data_set.addSample(in_data[i],in_data[i])

    #return the keys if we have
    if multiclass:
        return data_set,keys,out_data # a multiclass classifier
    else:
        return data_set

class_array = [5,10,13,26,52,104]
# for i in xrange(len(class_array)):
#     hidden_layers=[class_array[i]]
    #train the neural network ---------------------------------------------------------------------------------------------------------------------------------------------------
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.networks.rbm import Rbm
from pybrain.unsupervised.trainers.rbm import (RbmGibbsTrainerConfig,RbmBernoulliTrainer)
from pybrain.unsupervised.trainers import deepbelief
from pybrain.structure import TanhLayer,SoftmaxLayer

#filename ="mnist_testset_small.csv"
filename ="sparse_alphabet.csv"
data,keys,labels = loadCSV_auto(filename)
nn = buildNetwork(*([data.indim]+hidden_layers+[data.outdim]),hiddenclass=SoftmaxLayer)
trainer = BackpropTrainer(nn,data,learningrate=learning_rate,momentum=momentum,lrdecay=learning_decay,batchlearning=batch_learning)

#### Try Deep Learner #########
# cfg = RbmGibbsTrainerConfig()
# cfg.maxIter = 10
# rbm = Rbm.fromDims(5,1)
# trainer = RbmBernoulliTrainer(rbm, data, cfg)

total_errors = []
for runs in xrange(1,26,1):

    error = []
    validation_error = []
    print "Training...{0} with {1} hidden layers".format(filename,hidden_layers)

    if validation_proportion == 0. or validation_proportion == 0:
        for i in xrange(iterations):
            print "iteration {0} of {1} ".format(i, iterations)
            error.append(trainer.train())
    else:
        error,validation_error = trainer.trainUntilConvergence(validationProportion=validation_proportion)


    #Save the neural network
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    import pickle
    pickle.dump(nn, open('AutoNN.pkl','wb'))



    #Visualise the hidden units
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    from mpl_toolkits.mplot3d import Axes3D
    #import matplotlib.pyplot as plt
    import numpy as np

    print 'keys: {}'.format(keys)
    print 'labels count for keys: {}'.format([labels.count(k) for k in keys])

    c = ['r','y','g','c','b','k']
    all_activations = [np.array([0.]*hidden_layers[0]) for i in xrange(len(keys))]
    #go through all the data and add up occurrences:
    for i in xrange(len(labels)):
        nn.activate(data.getSample(i)[0])
        activations = nn['hidden0'].outputbuffer[nn['hidden0'].offset]
        #print activations
        #print labels[i]
        all_activations[keys.index(labels[i])] += activations

    #set up a figure

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in xrange(len(all_activations)):
        ys = all_activations[i]/labels.count(keys[i]) #nromalise occurrences
        xs =range(hidden_layers[0])
        zs = i+1

        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        #cs = [c] * len(xs)
        #cs[0] = 'c'
        ax.bar(xs, ys, zs, zdir='x',color=c[:len(xs)], alpha=0.8)

    plt.title('AutoEncoded: {0} for {1} epochs \r\n with {2} Hidden Layered ANN \r\n Activation'.format(filename,iterations,hidden_layers))
    ax.set_xlabel('Class')
    ax.set_ylabel('Hidden Unit')
    ax.set_zlabel('Activation')

    # plt.show()
    plt.savefig('AutoEncoded.{0}.{1}epochs.{2}".format(filename,iterations,hidden_layers).png')


    import numpy as np
    convertedlist = np.around(all_activations, decimals=8)

    #myFormattedList = [ '%.2f' % lem for elem in convertedlist ]

    for i in xrange(len(convertedlist)):
        print '{0}({1}):{2}'.format(keys[i],i,convertedlist[i])

    print 'error: {0}, validation error: {1}'.format(error,validation_error)


    plt.figure(2)
    plt.plot(error,'b')
    plt.plot(validation_error,'r')

    #label the axes
    plt.ylabel("Training Error")
    plt.xlabel("Training Steps")
    plt.title('AutoEncoded: {0} for {1} epochs \r\n with {2} Hidden Layered ANN \r\n Error Function'.format(filename,iterations,hidden_layers))

    #show the plot
    #plt.show()
    #save the plot
    plt.savefig('AutoEncoded.{0}.{1}epochs.{2}".format.err(filename,iterations,hidden_layers).png')

    h1+=1
    h2+=1
    hidden_layers = [h1,h2]
    total_error= 0
    for this_error in error:
        total_error+=this_error**2
    print 'total error is: {0}'.format(total_error)
    total_errors.append(total_error)

    plt.figure(3)
    plt.plot(total_errors,'b')
    plt.plot(runs,'r')

    #label the axes
    plt.ylabel("Total Error")
    plt.xlabel("Mirrored Double Hidden Layer Size")
    plt.title('AutoEncoded: {0} for {1} epochs \r\n with {2} Hidden Layered ANN \r\n Error Function'.format(filename,iterations,hidden_layers))

#show the plot
#plt.show()
#save the plot
    plt.savefig('AutoEncoded.{0}.{1}epochs.{2}".format.totalerr(filename,iterations,hidden_layers).png')

