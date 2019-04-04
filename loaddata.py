import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os

#!pip install hyperopt
#!pip install hyperas
from  keras.models import Sequential
from keras.utils import np_utils
from  keras.preprocessing.image import ImageDataGenerator
from  keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Input
from  keras.datasets import cifar10
from  keras import regularizers
from  keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.misc import toimage
#from hyperas.distributions import uniform, choice
#from hyperopt import Trials, STATUS_OK, tpe
#from hyperas import optim
import numpy as np
import os
fileloc='dataset'
trainsize=(0.8)
dirc = os.listdir(fileloc)
xdata = []
ydata = []
for label in dirc:
	for img in os.listdir(fileloc+"/"+label):
		xdata.append(np.array(Image.open(fileloc+"/"+label+"/"+img)))
		ydata.append(label)


X_train, X_test, Y_train, Y_test = train_test_split(xdata, ydata, test_size=1-trainsize)
X_train=np.array(X_train) 
Y_train=np.array(Y_train) 
X_test=np.array(X_test)
Y_test=np.array(Y_test)
print(Y_test)

input_shape = (50, 87)
batch_size = 128
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.2
weight_decay = 1e-4
num_labels=11

model = Sequential()
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(Flatten())
# dropout added as regularizer
model.add(Dropout(dropout))
# output layer is 10-dim one-hot vector
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification tasks
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
			  
model.fit(X_train, Y_train, epochs=20, batch_size=batch_size)			  

loss, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))