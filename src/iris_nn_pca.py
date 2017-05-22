#nn build taken from pybrain.org

#from sklearn import tree
#from sklearn import preprocessing
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
#from sklearn import cross_validation
import time
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

#import the iris dataset
data=np.loadtxt("..\\pca\\attribute_pca_nn.txt", delimiter=",", usecols = (0,1) )
target=np.genfromtxt('..\\iris.data', delimiter=",", dtype='str', usecols = (2))

# import testing data (iris.test)
import numpy as np 
#test_data=np.loadtxt("..\\iris.test", delimiter=",", usecols = (0,1,2,3) )
#test_cat=np.genfromtxt('..\\iris.test', delimiter=",", dtype='str', usecols = (4))


#convert target category to number on training data 
# iris_type=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
# target = []
# for l in target_cat:
	# #print l
	# #print iris_type.index(l)
	# target.append(iris_type.index(l))
# target=np.array(target)
	
#convert target category to number on test data 
# iris_type=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
# test_target = []
# for l in test_cat:
	# #print l
	# #print iris_type.index(l)
	# test_target.append(iris_type.index(l))
# test_target=np.array(test_target)

#setup the ClassificationDataSet dimensions with 4 attributes petal width, petal height, sepal width, sepal height and the target values 
ds = ClassificationDataSet(2, 1, class_labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
#alldata.setField('input', data)
#alldata.setField('target', target)

X = data
y = target
for i in range(len(X)):
 ds.addSample(int(X[i]),y[i])

#duplicates the original targets and stores them in an (integer) field named class. 

tstdata, trndata = ds.splitWithProportion( 0.25 )



tstTargets = tstdata.getField('target')
print tstTargets

ds._convertToOneOfMany( )
 
 #print out a data sample 
print "Number of training patterns: ", len(ds)
print "Input and output dimensions: ", ds.indim, ds.outdim
print "First sample (input, target, class):"
#print ds['input'][0], ds['target'][0], ds['class'][0]

#Now build a feed-forward network with 5 hidden units
fnn = buildNetwork( ds.indim, 5, ds.outdim, outclass=SoftmaxLayer )
print "activate on random weights"
print fnn.activate([5.1,3.5,1.4,0.2])
#fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

#Set up a back propagation trainer that takes the network and training dataset as input.
trainer = BackpropTrainer( fnn, dataset=ds, momentum=0.4, verbose=True, weightdecay=0.04)
#trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)


for i in range(20):
   trainer.trainEpochs( 5 )
   trnresult = percentError( trainer.testOnClassData(), ds['class'] )
   tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
   print "epoch: %4d" % trainer.totalepochs, "  train error: %5.4f%%" % trnresult, "  test error: %5.8f%%" % tstresult
   print "epoch: %4d" % trainer.totalepochs, "  train error: %5.4f%%" % trnresult 
   print "printint test on class data tst data "
   #tstresult = trainer.testOnClassData(dataset=tstdata)
   #tstresult = np.array(tstresult)
   #print "Accuracy on test set: %7.4f" % accuracy_score(tstTargets, tstresult, normalize=True)
   #print "percent error on test data "
   #print percentError( trainer.testOnClassData(dataset=tstdata ) )

print tstresult
print "activate on trained network"
print fnn.activate([5.1,3.5,1.4,0.2])
 
# visualize data
# taken from http://sujitpal.blogspot.com/2014/07/handwritten-digit-recognition-with.html
# get 100 rows of the input at random
# print "Visualize data..."
# idxs = np.random.randint(X.shape[0], size=100)
# fig, ax = plt.subplots(10, 10)
# img_size = math.sqrt(n_features)
# for i in range(10):
    # for j in range(10):
        # Xi = X[idxs[i * 10 + j], :].reshape(img_size, img_size).T
        # ax[i, j].set_axis_off()
        # ax[i, j].imshow(Xi, aspect="auto", cmap="gray")
# plt.show()