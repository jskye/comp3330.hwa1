from svmutil import svm_load_model,svm_predict
from svm import *
from numpy import arange,meshgrid,zeros
import matplotlib.pyplot as plt
from numpy import array

X = arange(-6.,6.,0.2)
Y = arange(-6.,6.,0.2)
X,Y = meshgrid(X,Y)
Z = zeros(X.shape)

filename = 'synthetic_control.data.csv_gamma150.0_c50.0.pk1'
model = svm_load_model(filename)


for i in range(len(X)):
    for j in range(len(Y)):
        #print svm_predict([0.],[[X[i][j],Y[i][j]]],model)[0][0]
        result = 20*int(svm_predict([0.],[[X[i][j],Y[i][j]]],model,'-q')[0][0])
        if result == 20:
            Z[i][j] = 20 #lower limit
        elif result ==40:
            Z[i][j] = 40 #lower limit
        elif result == 60:
            Z[i][j] = 60 #lower limit
        elif result ==80:
            Z[i][j] = 80 #lower limit
        else:
            Z[i][j] = 100 #higher limit






plt.imshow(Z)
plt.gcf()
plt.clim()
plt.title("SVM Activation:{0}".format(filename))


#plt.show()

plt.savefig('{0}.png'.format(filename), bbox_inches='tight')
