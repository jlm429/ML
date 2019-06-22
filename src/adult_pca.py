import numpy as np 
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import decomposition

#load data
filtered_data=np.array(np.genfromtxt('..\\filtered-data.txt', delimiter=',',autostrip=True))
scaled_filtered_data=preprocessing.scale(filtered_data)

#set the target class/attribute 
salary=scaled_filtered_data[:,14]
scaled_filtered_data=scaled_filtered_data[:,:-1]

#set the target class to -1 or 1 for values less than 50 k or greater than 50 k respectively 
salary2=np.array([-1]*len(salary))
for i in range(0,len(salary2)):
    if salary[i]>0:
        salary2[i]=1
		
#setup cross validation
train_samples, test_samples, train_out, test_out = cross_validation.train_test_split(scaled_filtered_data, salary2, test_size=0, random_state=0)
X = train_samples
y = train_out

#apply pca
pca = decomposition.PCA(n_components=2)
pca.fit(X)
new_X = pca.transform(X)

#save data
np.savetxt("attribute_pca_out.csv", new_X, delimiter=",")
np.savetxt("target_pca_out.csv", y, delimiter=",")