import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import datetime


iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target !=0) * 1

plt.figure(figsize = (10,6))
plt.scatter( X[y==0][:, 0], X[y==0][:,1], color='b', label='0')
plt.scatter( X[y==1][:, 0], X[y==1][:,1], color='r', label='1')
#plt.legend()
#plt.show()

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0],1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmod(self, z):
        return 1/ (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1-y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        #if self.fit_intercept:
            #X = self.__add_intercept(X)
        
        #weignts initialization 
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X,self.theta)
            h = self.__sigmod(z)
            gradient = np.dot( X.T, (h -y)) / y.size
            self.theta -= self.lr *gradient

            if(self.verbose == True and i % 100000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmod(z)
                print(f'loss: {self.__loss(h,y)} \t')
        
        print(self.theta)
    
    def predict_prob(self, X):
        #if self.fit_intercept:
           # X = self.__add_intercept(X)
        
        return self.__sigmod(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold

def main():
    model = LogisticRegression(lr=0.1, num_iter=300000)
    X_test, y_test, X_train, y_train = train_test_split(X, y, test_size=0.1, random_state=0)
    model.fit(X_train, y_train)

if __name__ == "__main__":
    main()
    
