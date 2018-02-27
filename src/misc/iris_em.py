print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.mixture import GMM


np.random.seed(42)


iris = datasets.load_iris()
data = scale(iris.data)

n_samples, n_features = data.shape
n_classes = len(np.unique(iris.target))
print "number of classes = "
print n_classes
labels = iris.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_classes, n_samples, n_features))




GMM(init='random', n_clusters=n_classes, n_init=4)
GMM.fit(data)