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
import time



filtered_data=np.array(np.genfromtxt('..\\filtered-data.txt', delimiter=',',autostrip=True))
scaled_filtered_data=preprocessing.scale(filtered_data)

#set the target class/attribute 
salary=scaled_filtered_data[:,14]
scaled_filtered_data=scaled_filtered_data[:,:-1]

print scaled_filtered_data.shape

#set the target class to -1 or 1 for values less than 50 k or greater than 50 k respectively 
salary2=np.array([-1]*len(salary))
for i in range(0,len(salary2)):
    if salary[i]>0:
        salary2[i]=1
		


train_samples, test_samples, train_out, test_out = cross_validation.train_test_split(scaled_filtered_data, salary, test_size=.5, random_state=0)

	
X=train_samples
y=train_out
# cvX=test_samples
# cvy=test_out


#setup the ClassificationDataSet dimensions with 4 attributes petal width, petal height, sepal width, sepal height and the target values 
ds = ClassificationDataSet(14, 1, class_labels=['>50K, <=50K'])
cvds = ClassificationDataSet(14, 1, class_labels=['>50K, <=50K'])

libsvm_start=time.time()
#add data to ClassificationDataSet
for i in range(len(X)):
 ds.addSample(X[i],y[i])
 
# #add data to CV ClassificationDataSet
# for i in range(len(cvX)):
 # cvds.addSample(cvX[i],cvy[i])
 
 
#split data then duplicates the original targets and stores them in an (integer) field named class.
tstdata, trndata = ds.splitWithProportion( 0.25 )
tstTargets = tstdata.getField('target')
ds._convertToOneOfMany( ) 

# cvTargets = cvds.getField('target')
# cvds._convertToOneOfMany( ) 




#print out a data sample 
print "Number of training patterns: ", len(ds)
print "Number of testing patterns: ", len(cvds)
print "Input and output dimensions: ", ds.indim, ds.outdim
print "First sample (input, target, class):"
print ds['input'][0], ds['target'][0], ds['class'][0]

#Now build a feed-forward network with 15 hidden units
fnn = buildNetwork( ds.indim, 5, ds.outdim, outclass=SoftmaxLayer )

#Set up a back propagation trainer that takes the network and training dataset as input.
trainer = BackpropTrainer( fnn, dataset=ds, momentum=0.4, verbose=True, weightdecay=0.04)


for i in range(20):
   trainer.trainEpochs( 5 )
   trnresult = percentError( trainer.testOnClassData(), ds['class'] )
   #trnresult = percentError( trainer.testOnClassData(dataset=trndata), trndata['class'] )
   #print "epoch: %4d" % trainer.totalepochs, "  train error: %5.4f%%" % trnresult, "  test error: %5.4f%%" % tstresult
   print "epoch: %4d" % trainer.totalepochs, "  train error: %5.4f%%" % trnresult
   tstresult = trainer.testOnClassData(dataset=tstdata)
   tstresult = np.array(tstresult)
   print "Accuracy on test set: %7.9f" % accuracy_score(tstTargets, tstresult, normalize=True)
 
libsvm_train_end=time.time()
print "Time Required for training                  : ",str(libsvm_train_end-libsvm_start)+" s"