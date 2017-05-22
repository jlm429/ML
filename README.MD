### Unsupervised Learning & Dimensionality Reduction 

> Data Sets Taken From UCI 

> <a href="https://archive.ics.uci.edu/ml/datasets/Iris">Iris Data Set</a>

> Target Classes: Iris-setosa,  Iris-virginica,  Iris-versicolor. One class is linearly separable from the other 2; the latter are not linearly separable from each other. 

> <a href="https://archive.ics.uci.edu/ml/datasets/Adult">Adult Data Set</a>. 

> Target Classes: <=50k, >50k.  The adult data set is more robust with 15 attributes and 45,000 total data items.   The prediction task is to determine whether a person would make over $50,000 a year using the 14 attributes listed.  

### Dimensionality Reduction

#### Principle Component Analysis (<a href="https://github.com/jlm429/UnsupervisedLearning/blob/master/src/iris_pca.py">src/iris_pca.py</a>, <a href="https://github.com/jlm429/UnsupervisedLearning/blob/master/src/adult_pca.py">src/adult_pca.py</a>) 

> Coding Resource: <a href="http://scikit-learn.org/stable/"> http://scikit-learn.org/</a>.  

> Scikit-learn provided some nice visuals to help determine which attributes gave the most variance.  After applying a linear transformation to the iris data set, I plotted the data points which made it easier to pin down a separating line for the Iris-virginica and Iris-versicolor classes (iris\pca\iris_pca.py).  For the Adult Data Set, PCA ranked attribute importance similar to the decision tree classifier using a GINI index.  From the plot of eigenvalues on the adult data set, the values drop off after the first few attributes indicating only the first few attributes will be needed.   Listed below are the PCA ranked attributes, and distribution of eigenvectors for both data sets and the 3D data plot for the iris data set.  

#### Iris Data Set PCA 3D Data Plot

![Component Diagram](https://github.com/jlm429/UnsupervisedLearning/blob/master/images/IRISPCAplot.png)

#### Adult Data Set PCA Distribution of Eigenvalues (Y=Value, X=Attribute)

![Component Diagram](https://github.com/jlm429/UnsupervisedLearning/blob/master/images/ADULTPCAEIGENVALUES.png)

#### Adult Data Set PCA Ranked attributes:

![Component Diagram](https://github.com/jlm429/UnsupervisedLearning/blob/master/images/ADULTPCARANKED.PNG)


#### Independent Component Analysis (<a href="https://github.com/jlm429/UnsupervisedLearning/blob/master/src/iris_ica.py">src/iris_ica.py</a>, <a href="https://github.com/jlm429/UnsupervisedLearning/blob/master/src/adult_ica.py">src/adult_ica.py</a>)


>The scikit learn ICA algorithm uses kurtosis values to determine attribute independence.  Listed below are the kurtosis for each attribute and graphs for attribute values with high kurtosis after applying the ICA transformation.   


![Component Diagram](https://github.com/jlm429/UnsupervisedLearning/blob/master/images/ICACHARTS.PNG)


### Neural Network with Dimensionality Reduction (<a href="https://github.com/jlm429/UnsupervisedLearning/blob/master/src/adult_nn.py">src/adult_nn.py</a>)

>I re-ran reduced data sets through a neural network and was able to achieve results similar to the original data set with much faster running times.  I also used a decision tree classifier to verify results and was again able to accurately classify the data after applying ICA, PCA, or RPA using only two of the original 14 attributes.  Listed below are accuracy scores and running times for the neural network using ICA, PCA, RPA and Variance Threshold feature selection algorithms on the adult data set.   

![Component Diagram](https://github.com/jlm429/UnsupervisedLearning/blob/master/images/NNCharts.PNG)


