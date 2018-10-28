import numpy as np 
from matplotlib import pyplot as plt

X = np.array([
        [-2, 4, -1],
        [4, 1, -1], 
        [1, 6, -1], 
        [2, 4, -1], 
        [6, 2, -1]
    ])

y = np.array([-1, -1, 1, 1, 1])

for d, sample in enumerate(X):
    # Plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Print a possible hyperplane, that is seperating the two classes.
#plt.plot([-2,6],[6,0.5])
#plt.show()

#using sgd to minimize our function 
def svm_sgd(X,Y):
    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 100000

    for epoch in range(1,epochs):
        for i, x in enumerate(X):
            if(Y[i]*np.dot(X[i], w)) < 1:
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1/epoch) *w))
            else:
                w = w + eta *(-2 *(1/epoch) * w)

    return w

def svm_sgd_plot(X,Y):
    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 100000
    errors = []

    for epoch in range(1,epochs):
        error = 0
        for i, x in enumerate(X):
            if(Y[i]*np.dot(X[i], w)) < 1:
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1/epoch) *w))
                error = 1
            else:
                w = w + eta *(-2 *(1/epoch) * w)
        errors.append(error)
    
    plt.plot(errors, '|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()

    
#print(svm_sgd(X,y))
#svm_sgd_plot(X,y)

#evaluation with our test sample 
for d, sample in enumerate(X):
    # Plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

#add our test samples
plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

w = svm_sgd(X,y)
#print the hyperplane calculated b svm_sgd()
x2=[w[0], w[1], -w[1], w[0]]
x3=[w[0],w[1], w[1], -w[0]]

x2x3 = np.array([x2, x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')

plt.show()