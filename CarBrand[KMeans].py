#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 21:24:11 2019

@author: hyacinth
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("cars.csv")

X = dataset[dataset.columns[:-1]]
X = X.convert_objects(convert_numeric=True)
X.head()

# Eliminating Null values
for i in X.columns:
    X[i] = X[i].fillna(int(X[i].mean()))
    

# Using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of cllusters')
plt.ylabel('WCSS')
plt.show()

# Applying K-means to the cars dataset

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=10)
y_kmeans = kmeans.fit_predict(X)
X = X.as_matrix(columns=None)

plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label='US Made')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label='Europe Made')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label='Japan Made')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='centroids')
plt.title('Clusters of car made')
plt.legend
plt.show()