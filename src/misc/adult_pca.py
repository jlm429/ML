#data formatting partially taken from ipython.org



import numpy as np 
from sklearn import preprocessing
from sklearn import cross_validation


from sklearn import decomposition
from sklearn import random_projection
import matplotlib.pyplot as plt
from scipy import signal
import time

#load the data
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
		

train_samples, test_samples, train_out, test_out = cross_validation.train_test_split(scaled_filtered_data, salary2, test_size=0, random_state=0)



X = train_samples
y = train_out


print X.shape
pca = decomposition.PCA(n_components=2)
pca.fit(X)
new_X = pca.transform(X)
print new_X.shape


np.savetxt("attribute_pca_out.csv", new_X, delimiter=",")
np.savetxt("target_pca_out.csv", y, delimiter=",")