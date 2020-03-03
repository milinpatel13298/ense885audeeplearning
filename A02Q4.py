IMG_SIZE=28

import tensorflow
import numpy
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import math	

def convert(x):			#converts the label datasets into one-hot encoded format
    y=numpy.zeros([len(x),10])
    z=numpy.eye(10)
    for i in range(len(x)):
        y[i]=(z[(x[i])])
    return y

def softMax(list):
	sum=0
	for i in range(len(list)):
		sum+=math.exp(list[i])
	for i in range(len(list)):
		list[i]=math.exp(list[i])/sum
	return list

def stochasticGradientDescent(X_train, Y_train_new, alpha,penalty):
	w=numpy.zeros((10,784))
	bias=numpy.zeros(10)
	print("Starting processing each image one by one")
	for i in range(len(X_train)):
		y_predicted=softMax(numpy.dot(w,X_train[i])+bias)
		loss_gradient=y_predicted-Y_train_new[i]
		w-=(2*alpha*loss_gradient.reshape(10,1)*X_train[i]+(2*penalty*w))
		bias-=(2*alpha*loss_gradient+(2*penalty*bias))
		if i==10000:
			print("Completed processing 10000 images")
		if i==20000:
			print("Completed processing 20000 images")
		if i==30000:
			print("Completed processing 30000 images")
		if i==40000:
			print("Completed processing 40000 images")
		if i==50000:
			print("Completed processing 50000 images")
		if i==59999:
			print("Completed processing 60000 images")
	return w,bias

with open('train-images-idx3-ubyte.gz', 'rb') as f:
	X_train = extract_images(f)
with open('train-labels-idx1-ubyte.gz', 'rb') as f:
	Y_train = extract_labels(f)
with open('t10k-images-idx3-ubyte.gz', 'rb') as f:
	x_test = extract_images(f)
with open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
	y_test = extract_labels(f)

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
x_test=x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')
Y_train_new=convert(Y_train)
y_test_new=convert(y_test)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
x_test = x_test / 255

#learned_weights=numpy.zeros((10,784))
learned_weights,learned_bias=stochasticGradientDescent(X_train,Y_train_new,alpha=0.1,penalty=0.001)
print(learned_weights)
print(learned_weights.shape)
success=0
confusion_matrix=[[0 for _ in range(10)] for _ in range(10)]
for i in range(len(x_test)):
	y_predicted=numpy.argmax(softMax(numpy.dot(learned_weights,x_test[i])+learned_bias))
	confusion_matrix[y_test[i]][y_predicted]+=1
	if y_predicted==y_test[i]:
		success+=1
	else:
		print(y_test[i],"was wrongly predicted as",y_predicted)
print("#############")
print("accuracy of model",success/10000)
print("#############")

s = [[str(e) for e in row] for row in confusion_matrix]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]
print('\n'.join(table))
print("##########################################")
