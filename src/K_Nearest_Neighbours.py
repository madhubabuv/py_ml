#!/usr/bin/env python


import numpy as np
from scipy.stats import itemfreq


class KNN:

	def __init__(self,train_file,test_file,k):

		self.train_data=np.loadtxt(train_file,delimiter=',')

		self.test_data=np.loadtxt(test_file,delimiter=',')

		self.test_x=self.test_data[:,range(self.test_data.shape[1]-1)]
		
		self.test_y=np.atleast_2d(self.test_data[:,self.test_data.shape[1]-1]).reshape(-1,1)

		self.train_x=self.train_data[:,range(self.train_data.shape[1]-1)]
	
		self.train_y=(self.train_data[:,self.train_data.shape[1]-1]).reshape(-1,1)

		self.k=k

		self.normalize() #normalization x-x_min/(x_max-x_min)


	def normalize(self):


		x_min=np.amin(self.train_x,axis=0)

		x_max=np.amax(self.train_x,axis=0)

		diff=x_max-x_min

		temp=self.train_x/diff

		self.train_x=(self.train_x-np.ones_like(self.train_x)*x_min)/diff
	
		self.test_x=(self.test_x-np.ones_like(self.test_x)*x_min)/diff	

		
	def E_distance(self):

		diff= (self.train_x-np.ones_like(self.train_x)*self.sample)**2

		self.euclidean_distance=np.sum(diff,axis=1).reshape(-1,1)

		#print self.euclidean_distance

	def sort(self):
	
		self.E_distance()

		temp=np.append(self.euclidean_distance,self.train_y,axis=1)
	
		self.sorted_data=temp[temp[:,0].argsort()]


	def check_neighbours(self):
		
		self.sort()
		
		neighbours=self.sorted_data[:self.k]

		neighbour_labels=neighbours[:,1]

		#print neighbour_labels

		return itemfreq(neighbour_labels)
		
		
	
	def predict(self,sample):

		self.sample=sample

		label_count=self.check_neighbours()

		return label_count[label_count[:,1].argsort()[-1]][0]

	def accuracy_score(self,pred):

		return 1-sum(abs(pred-self.test_y))/len(pred)


if __name__=="__main__":


#/home/madhu/Desktop/py_ml/data/knn_data/pima-indians-diabetes-test.data

	train_data="../data/knn_data/pima-indians-diabetes-train.data"

	test_data="../data/knn_data/pima-indians-diabetes-test.data"

	obj=KNN(train_data,test_data,20)




	score=[]
	k_range=50
	for k in range(1,k_range):
		obj.k=k
		pred=[]
		for sample in obj.test_x:

			pred.append(obj.predict(sample))

		pred=np.reshape(pred,(-1,1))

		score.append(obj.accuracy_score(pred))


	print "The best K:",score.index(max(score))


	from matplotlib import pyplot as plt


	plt.plot(range(1,k_range),score)
	plt.legend(["$accuracy$"],loc='upper right')
	plt.xlabel("$K$")
	plt.ylabel("$accuracy$")
	plt.grid()
	plt.show()

#obj.sklearn()
	




