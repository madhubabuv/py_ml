#!/usr/bin/env python


import numpy as np

from matplotlib import pyplot as plt


class LienarRegression:

	def __init__(self,file_location,learning_rate=0.01,iterations=1000):

		self.learning_rate=learning_rate

		self.file_location=file_location

		self.iterations=iterations



	def ModelData(self):

		file_data=np.loadtxt(self.file_location,delimiter=',')
		
		y=file_data[:,1].reshape(-1,1)
		x=file_data[:,0].reshape(-1,1)

		return x,y

	def calcHypo(self,theta,x):

		return np.dot(x,theta).reshape(-1,1)

	def fit(self,x,y):

		x=np.append(np.ones_like(y),x,axis=1)

		theta=np.zeros_like(x[0])
		
		for  i in range(self.iterations):

			h=self.calcHypo(theta,x)

			grad=sum((h-y)*x)/len(x)

			theta-=self.learning_rate*grad

		self.theta=theta


	def predict(self,x):

		x=np.append(np.ones_like(x),x,axis=1)

		return np.dot(x,self.theta)
			
	
	def plot(self,x,y,y_):

		
		plt.scatter(x,y,marker='s',color='b',label='data')
		plt.plot(x,y_,color='g',label='line of best fit')

		plt.legend()

		plt.grid()

		plt.show()



	
		

file_loc="../data/LinearRegression.txt"

obj=LienarRegression(file_loc)

x,y=obj.ModelData()

obj.fit(x,y)

y_=obj.predict(x)

obj.plot(x,y,y_)



		
