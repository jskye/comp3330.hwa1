import pickle
from numpy import arange,meshgrid,zeros
import matplotlib.pyplot as plt
from matplotlib import colors



X = arange(-6.,6.,0.2)
Y = arange(-6.,6.,0.2)
X,Y = meshgrid(X,Y)
Z = zeros(X.shape)

picklejar = []
filename = 'test.results.5Spirals/SVM/5spirals.csv_gamma200.0_c50.0.pk1'
picklejar.append(filename)


for p in range(len(picklejar)):

    model = pickle.load(open(picklejar[p]))

    for i in range(len(X)):
        for j in range(len(Y)):
            result = model.activate([X[i][j],Y[i][j]])
            result_copy =result
            b = 20
            for z in range(0,4):

            # loop through finding smallest result and setting to next class cutoff
                if min(result) ==result_copy[0]:
                    Z[i][j] = b #lower limit
                    result_copy[0]=1000
                elif min(result) ==result_copy[1]:
                    Z[i][j] = b
                    result_copy[1]=1000
                elif min(result) ==result_copy[2]:
                    Z[i][j] = b
                    result_copy[2]=1000
                elif min(result) ==result_copy[3]:
                    Z[i][j] = b
                    result_copy[3]=1000
                else :
                    Z[i][j] = b
                    result_copy[4]=1000
                b+=20




            if result[0] > result[1]:
                Z[i][j] = 20 #lower limit
            elif result[1]>result[2]:
                Z[i][j] = 40
            elif result[2]>result[3]:
                Z[i][j] = 60
            elif result[3]>result[4]:
                Z[i][j] = 80
            else :
                Z[i][j] = 100




    #cm = plt.cm.get_cmap('RdYlBu')
    # make a color map of fixed colors
    # colormap = colors.ListedColormap(['red', 'green', 'blue', 'yellow', 'orange'])
    # bounds=[0,5,10,15,20]
    # norm = colors.BoundaryNorm(bounds, colormap.N)

    #plt.imshow(Z, cmap=colormap)
    plt.imshow(Z)

    plt.gcf()
    plt.clim()
    #plt.contourf(X, Y, Z)

    plt.title("{0}".format(filename))

    plt.savefig('{0}.png'.format(filename), bbox_inches='tight')
    plt.show()

