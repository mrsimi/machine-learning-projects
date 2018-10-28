import math
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import datetime
from sklearn.cross_validation import train_test_split

plt.style.use('ggplot')

#optimization technique: gradient descent 

class CustomLogisticRegression:
    
    def __init__(self, x, y, tolerance=0.00001):
        self.tolerance = tolerance
        self.cost =[]
        self.alpha= 0.1
        self.lambd=0.25
        self.iter = 2500
        self.x = x
        self.y = y

        #initialize theta 
        self.theta = np.random.rand(x.shape[0], 1)
    
    #the cost function 
    def cost_fn(self, m):
        h = self.sigmod_function(np.dot(self.x, self.theta)) #get the hypothesis 
        J = (1.0/m) * np.sum(-self.y*(np.log(h)) - (1.0 - self.y) * (np.log(1.0 - h))) 
        return J
    
    def sigmod_function(self,z):
        return 1.0/(1.0  + math.e**(-1*z))

    #gradient function 
    def gradients(self,m): 
        zrd = self.theta
        zrd[0,:] = 0
        h = self.sigmod_function(np.dot(self.x,self.theta))
        return ( 1.0/m ) * np.dot(self.x.T, ( h - self.y ) ) + (float(self.lambd)/m) * zrd 
    #batch gradient decent 
    def descent(self):
        for i in range(self.iter):
            self.cost.append(self.cost_fn(self.x.shape[0]))
            gradientz = self.gradients(self.x.shape[0]) 

            #update the theta based on the gradients 
            self.theta[0,:] = gradientz[0, :] - self.alpha * gradientz[0, :]
            self.theta[1,:] = gradientz[1, :] - self.alpha * gradientz[1, :]
        
        pred = np.dot(self.x, self.theta)
        pred[ pred >= 0.5] = 1
        pred[ pred < 0.5] = 0
    
def main():
    df = pd.read_table('logistic_regression_data.txt', sep=',', names=('featureOne', 'featureTwo', 'label'))
    y = np.array(df['label']).T
    df = np.array(df)
    x = df[:,:2] 

    #normalize the data 
    df = (df - df.mean())/ (df.max() - df.min())

    x_test, y_test, x_train, y_train = train_test_split(x, y, test_size=0.1, random_state=0)

    glm = CustomLogisticRegression(x_train, y_train)
    glm.descent()
    plt.scatter(x[:, 0], y)

    plt.show()

if __name__ == "__main__":
    main()