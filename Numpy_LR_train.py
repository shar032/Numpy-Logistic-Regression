import pandas as pd
from oop_Numpy_Logistic_Regression import LogisticRegression

# import dataset
dataset = pd.read_csv('PerovNonPerovFINALANN.csv')
# list labels
y = dataset.iloc[0:622,0].values
# parse inputs
for i in range(1,10):
    vars()['x' + str(i)] = dataset.iloc[0:622, i].values
    
# create input matrix
X = [[1, x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i], x8[i], x9[i]] for i in range(622)]

# define hyperparameters
m = len(x1)
lmbda = 0.07
lr = 0.1
num_epochs = 200
validation_size = 70 

# instantiate model
LR = LogisticRegression(num_epochs, lr, lmbda, X, y, validation_size, m)    

# train model            
LR.train(show_plots = True) 