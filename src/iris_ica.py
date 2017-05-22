print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Compute ICA
print X.shape

ica = FastICA(n_components=4)
S_ = ica.fit_transform(X)  # Reconstruct signals
print S_.shape
print S_

A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=4)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

###############################################################################
# Plot results

plt.figure()

models = [X, S_]
names = ['Observations (mixed signal)',
         'ICA recovered signals']
colors = ['red', 'steelblue', 'orange', 'yellow', 'green']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()