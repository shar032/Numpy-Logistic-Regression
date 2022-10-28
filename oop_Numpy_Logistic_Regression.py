#Regularised Logistic Regression for Perov/Non-Perov Classification

import matplotlib.pyplot as plt
import numpy as np
import random


""" Define Logistic Regression Class """
class LogisticRegression:
    def __init__(self, num_epochs, lr, lmbda, X, y, validation_size, m):
        self.num_epochs = num_epochs
        self.lr = lr
        self.lmbda = lmbda
        self.X = X
        self.y = y
        self.validation_size = validation_size
        self.validation_idx = random.sample(range(0, len(y)), self.validation_size)
        self.y_train = [y[i] for i in range(len(y)) if i not in self.validation_idx]
        self.X_train = [X[i] for i in range(len(X)) if i not in self.validation_idx]
        self.y_validation = [y[i] for i in range(len(y)) if i in self.validation_idx]
        self.X_validation = [X[i] for i in range(len(X)) if i in self.validation_idx]
        self.m = m
        self.theta = np.random.normal(0, 0.5, len(self.X[1]))

    def __sigmoid(self, theta, x):
        """ Sigmoid Function
            
        Args:
            theta (numpy.ndarray): weights vector
            x (numpy.ndarray): input vector 
                
        Returns:
            (float): sigmid activation output"""
            
        return 1/(1 + np.exp(-np.dot(theta,x)))
        
    def __single_cost(self, htheta, y):
        """ Binary Cross Entropy Cost for Predicted Data
            
        Args:
            htheta (float): Prediction
            y (float): label
                
        Returns:
            (float): binary cross entropy cost """
                
        return -y*np.log(htheta)-(1-y)*np.log(1-htheta)
        
    def __total_cost(self, theta):
        """ Total Cost Calculator For Training Set
                
        Args:
            theta (numpy.ndarray): weights vector
                    
        Returns:
            (float): Total Binary Cross Entropy Cost For Training Set """
            
        htheta_all = [self.__sigmoid(theta, self.X_train[i]) for i in range(len(self.X_train))]
        return (1/self.m)*sum([self.__single_cost(htheta_all[i], self.y_train[i]) for i in range(len(htheta_all))])
        
        
    def __percent_accuracy_training(self, theta):
        """ Calculates Percet Accuracy for Training Data
            
        Args:
            theta (numpy.ndarray): weights vector
                    
        Returns:
            (float): Percentage Accuracy of Prediction """
                    
        htheta_all = [round(pred) for pred in [self.__sigmoid(theta,self.X_train[i]) for i in range(len(self.X_train))]]
        return len([i for i in range(len(htheta_all)) if htheta_all[i] == self.y_train[i]])/len(self.y_train)
        
    def __percent_accuracy_validation(self, theta):
        """ Calculates Percet Accuracy for Validation Data
            
        Args:
            theta (numpy.ndarray): weights vector
                
        Returns:
            (float): Percentage Accuracy of Prediction """
            
        htheta_all = [round(pred) for pred in [self.__sigmoid(theta,self.X_validation[i]) for i in range(len(self.X_validation))]]
        return len([i for i in range(len(htheta_all)) if htheta_all[i] == self.y_validation[i]])/len(self.y_validation)
        

    def train(self, show_plots):
        """ Regularized Logistic Regression training for binary classification of perovskite materials using
            batch gradient descent 
        
        Args:
            show_plots (bool): To show plots at end of training
            
        Returns:
                Prints training metrics and shows plots"""
            
        #theta = np.random.normal(0, 0.5, len(self.X[1]))
        all_epochs, trainset_costs, training_accuracies, validation_accuracies = [[] for _ in range(4)]
        
        for epoch in range(self.num_epochs):
            dtheta = list(np.zeros(len(self.theta)))
                
            for j in range(len(dtheta)):
                dtheta[j] = -(self.lr/self.m)*sum([(self.__sigmoid(self.theta, self.X_train[i])-self.y_train[i])*self.X_train[i][j] for i 
                          in range(len(self.y_train))])-(self.lmbda/self.m)*self.theta[j]
                
            for k in range(len(self.theta)):
                self.theta[k] += dtheta[k]
                
            all_epochs.append(epoch)
            print("Epoch: {}".format(epoch))
            trainset_costs.append(self.__total_cost(self.theta))
            print("Total Cost : {}".format(self.__total_cost(self.theta)))
            training_accuracies.append(self.__percent_accuracy_training(self.theta))
            print("Percent Training Accuracy: {}".format(self.__percent_accuracy_training(self.theta)))
            validation_accuracies.append(self.__percent_accuracy_validation(self.theta))
            print("Percent Validation Accuracy: {}".format(self.__percent_accuracy_validation(self.theta)))
            
        
        if show_plots:    
            """ Plot training metrics over epoch """
            
            plt.plot(all_epochs, training_accuracies)
            plt.plot(all_epochs, validation_accuracies)
            plt.legend(["Training set accuacy", "Validation set accuracy"])
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Train and Validation Set Accuracy (%) vs. Epoch")
        
    def predict(self, x, prob):
        """ Predict binary class of new input data
            
        Args:
            x (array/list): input vector
            prob (bool): return probability or rounded label
            
        Returns:
            (float): label or probability """
                
        out = self.__sigmoid(self.theta, x) if prob else float(round(self.__sigmoid(self.theta, x)))
        return out 

        


