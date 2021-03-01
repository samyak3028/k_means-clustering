# k_means-clustering

#### Simple k-means clustering (centroid-based) using Python

## Code Requirements

#### Python 3.5
#### Numpy 1.11.0

## Description
#### k-Means clustering is one of the most popular clustering methods in data mining and also in unsupervised machine learning.In this area of machine learning,we will have the data, but we don’t have any target variable as in the case of supervised learning. So the goal here is to observe the hidden patterns among the data and group them into clusters

![Before Clustering](https://github.com/samyak3028/k_means-clustering/blob/main/before.png?raw=true)
##### Before clustering

![After Clustering](https://github.com/samyak3028/k_means-clustering/blob/main/after1.png?raw=true)
##### After clustering


##### Firstly we have to intitilize cluster centre for each cluster.

##### Secondly for each data point we need to take eucledian distance euclidian distance from each point to all the centroids and store in a m X K matrix. So every row in EuclidianDistance matrix will have distances of that particular data point from all the centroids.

##### Thirdly we shall find the minimum distance and store the index of the column in a vector C and assign cluster based on minimal distance to all centroid.We need to regroup the data points based on the cluster index  and store in the Output dictionary and also compute the mean of separated clusters and assign it as new centroids. Y is a temporary dictionary which stores the solution for one particular iteration.

##### Fourthly adjust the centroids on basis of average of all the data which was computed in previous step.

##### At last repeat this process till cluster are well seperated.
