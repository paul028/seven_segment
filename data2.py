import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os

def load_dataset(fileloc='dataset',trainsize=(0.8)):
	"""
	Load image dataset from different folders representing each class.
	Args:
		fileloc: file location of the main folder
		trainsize: training split percent

	Return:
		X_train: (x,h,w) array with x number of hxw sample images for training
		Y_train: (x,) array of output labels for training
		X_test: (x,h,w) array with x number of hxw sample images for testing
		Y_test: (x,) array of output labels for testing

	"""

	dirc = os.listdir(fileloc)
	xdata = []
	ydata = []
	for label in dirc:
		for img in os.listdir(fileloc+"/"+label):
			asd = np.array(Image.open(fileloc+"/"+label+"/"+img))
			xdata.append(np.array(Image.open(fileloc+"/"+label+"/"+img)))
			ydata.append(label)
			print(asd.shape)
			if(len(asd.shape)==3):
				print("Error")

	X_train, X_test, Y_train, Y_test = train_test_split(xdata, ydata, test_size=1-trainsize)
	return (np.array(X_train), np.array(Y_train)), (np.array(X_test), np.array(Y_test))
	
if __name__ =="__main__":
	(X_train, Y_train), (X_test, Y_test) = load_dataset()
	print("X train shape: ", X_train.shape)
	print("X train img shape: ", X_train[0].shape)
	print("Y train shape: ", Y_train)
	print("Y train img shape: ", Y_train[0])

