import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(figsize = (14,6), dpi = 80)

df = sns.load_dataset('tips')
print(df.info())
col = df.columns
#import required modules
import numpy as np

def Factorize(d):
    dfac = d.copy()
    for i in col:
        if dfac[i].dtype.name == 'category':
            dfac[i] = pd.factorize(dfac[i])[0]
    d = dfac
    return d

def DummiesNoDropFirst(d):
    ddum = d.copy()     
    for i in col:
        if ddum[i].dtype.name == 'category':
            k = pd.get_dummies(ddum[i], dtype = int)
            ddum = pd.concat([ddum, k], axis = 1)
            del ddum[i]
    d = ddum
    print(d.info())
    return d


class LogisticRegression:
    def __init__(self,x,y):      
        self.intercept = np.ones((x.shape[0], 1))  
        self.x = np.concatenate((self.intercept, x), axis=1)
        self.weight = np.ones(self.x.shape[1])
        self.y = y
        
    #Sigmoid method
    def sigmoid(self, x, weight):
        z = np.dot(x, weight)
        return 1 / (1 + np.exp(-z))
    
    #method to calculate the Loss
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h))
    
    #Method for calculating the gradients
    def gradient_descent(self, X, h, y):
        return  np.dot(X.T, h) - np.dot(X.T, y)

    
    def fit(self, lr, iterations):
        for i in range(iterations):
            sigma = self.sigmoid(self.x, self.weight)
            print(self.weight)
            #loss = self.loss(sigma,self.y)

            dW = self.gradient_descent(self.x , sigma, self.y)
            
            #Updating the weights
            self.weight -= lr * dW
            print(self.weight)

        return print('fitted successfully to data', self.weight)
    
    #Method to predict the class label.
    def predict(self, x , treshold):
        x = np.concatenate((self.intercept, x), axis=1)
        result = self.sigmoid(x, self.weight)
        result = (result >= treshold)
        y_pred = np.zeros(result.shape)
        for i in range(len(y_pred)):
            if result[i] == True: 
                y_pred[i] = 1
            else:
                continue
                
        return y_pred
            



#Loading the data


y = df.iloc[:, -1]
#print(data)
x = DummiesNoDropFirst(df.iloc[:,0:6])

print(y)
#creating the class Object
regressor = LogisticRegression(x,y)

regressor.fit(0.1 , 3)

y_pred = regressor.predict(x,0.5)

print('accuracy -> {}'.format(sum(y_pred == y) / y.shape[0]))