#!/usr/bin/env python


import numpy as np

from matplotlib import pyplot as plt


class LogisticRegression:

	def __init__(self,file_location,learning_rate=0.0005,iterations=50000):

		self.learning_rate=learning_rate

		self.file_location=file_location

		self.iterations=iterations

		self.sigmoid=lambda x:(1/(1+np.exp(-x)))



	def ModelData(self):

		file_data=np.loadtxt(self.file_location,delimiter=',')
		
		y=file_data[:,2].reshape(-1,1)

		x=file_data.T[:-1].T

		return x,y

	def like_fit(self,x,y): # using maximum likelihood


		x=np.append(np.ones_like(y),x,axis=1)
		self.theta = np.zeros(x.shape[1])

		for step in xrange(self.iterations):

			scores = np.dot(x, self.theta)
			predictions = self.sigmoid(scores).reshape(-1,1)

			# Update weights with gradient
			output_error_signal = y - predictions
			gradient = np.dot(x.T, output_error_signal).ravel()
			
			self.theta += self.learning_rate * gradient

        #self.theta=theta


	def calcHypo(self,theta,x):

		return np.dot(x,theta).reshape(-1,1)

	def grad_fit(self,x,y): # using gradient descent

		x=np.append(np.ones_like(y),x,axis=1)


		theta=np.ones_like(x[0])
		
		for  i in range(self.iterations):

			h=self.sigmoid(self.calcHypo(theta,x))

			grad=sum((h-y)*x)/len(x)

			theta-=self.learning_rate*grad

		self.theta=theta


	def predict(self,x):

		x=np.append(np.ones(len(x)).reshape(-1,1),x,axis=1)

		return self.sigmoid(np.dot(x,self.theta))


		
	
		

file_loc="../data/LogisticRegression.txt"

obj=LogisticRegression(file_loc)

x,y=obj.ModelData()

obj.like_fit(x,y)

y_=obj.predict(x)

print obj.predict([[78.63542434898018,96.64742716885644]])




		
