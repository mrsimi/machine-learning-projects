from copy import deepcopy
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

plt.figure(figsize=(16,9))
plt.style.use('ggplot')

#import dataset 
data = pd.read_csv('k_means_data.csv')
#print("input data and shape")
#print(data.shape)
#print(data.head())

#all centroids used

#get values to plot 
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1,f2)))
plt.scatter(f1, f2, c='black', s=7)
#plt.show()

#euclidean distance 
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

k = 3   #number of clusters 
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X)-20, size=k)
#y coordinates of random centroids 
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

print("initial centroids: {0}".format(C))

#plotting along with the centroids 
plt.scatter(f1, f2, c="#050505", s=7)
plt.scatter(C_x, C_y, marker="*", s=200, c='g')
#plt.show()


C_old = np.zeros(C.shape)

clusters = np.zeros(len(X))

error = dist(C, C_old, None)

while error != 0:
    #assign each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    
    #storing the old centroid values 
    C_old = deepcopy(C)
    
    #finding the new centroids by taking the average value 
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    
    error = dist(C, C_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
#plt.show()
print(C)

