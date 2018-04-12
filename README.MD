### Dimensionality Reduction 

> Data Sets Taken From UCI 

> <a href="https://archive.ics.uci.edu/ml/datasets/Iris">Iris Data Set</a>

> Target Classes: Iris-setosa,  Iris-virginica,  Iris-versicolor. One class is linearly separable from the other 2; the latter are not linearly separable from each other. 

> <a href="https://archive.ics.uci.edu/ml/datasets/Adult">Adult Data Set</a>. 

> Target Classes: <=50k, >50k.  The prediction task is to determine whether a person would make over $50,000 a year using the 14 attributes listed.  

#### Principle Component Analysis (<a href="https://github.com/jlm429/UnsupervisedLearning/blob/master/src/iris_pca.py">src/misc/iris_pca.py</a>, <a href="https://github.com/jlm429/UnsupervisedLearning/blob/master/src/adult_pca.py">src/misc/adult_pca.py</a>) 

> Coding Resource: <a href="http://scikit-learn.org/stable/"> http://scikit-learn.org/</a>.  

> Apply a linear transformation to separate the data (iris\pca\iris_pca.py).  Adult Data Set attributes are ranked using principal component analysis for dimensionality reduction.  Eigenvalues decrease dramatically after the first few attributes suggesting those needed in order to accurately classifying data.   Listed below are the PCA ranked attributes and distribution of eigenvectors for both data sets and a 3D data plot of the iris data (graphed using matplotlib).  

#### Iris Data Set PCA 3D Data Plot

![Component Diagram](https://github.com/jlm429/UnsupervisedLearning/blob/master/images/IRISPCAplot.png)

#### Adult Data Set PCA Ranked attributes:

![Component Diagram](https://github.com/jlm429/UnsupervisedLearning/blob/master/images/ADULTPCARANKED.PNG)

### Neural Network with Dimensionality Reduction (<a href="https://github.com/jlm429/UnsupervisedLearning/blob/master/src/adult_nn.py">src/misc/adult_nn.py</a>)

>Running data sets through a neural network after dimensionality reduction gives similar results with faster running times.   

![Component Diagram](https://github.com/jlm429/UnsupervisedLearning/blob/master/images/NNCharts.PNG)


