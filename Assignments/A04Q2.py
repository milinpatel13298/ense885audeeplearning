IMG_SIZE=28
import tensorflow
from tensorflow import keras
import numpy
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import matplotlib.pyplot as plt


def convert(x):
    y=numpy.zeros([len(x),10])
    z=numpy.eye(10)
    for i in range(len(x)):
        y[i]=(z[(x[i])])
    return y


with open('train-images-idx3-ubyte.gz', 'rb') as f:
	train_images = extract_images(f)
with open('train-labels-idx1-ubyte.gz', 'rb') as f:
	train_labels = extract_labels(f)

with open('t10k-images-idx3-ubyte.gz', 'rb') as f:
	test_images = extract_images(f)
with open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
	test_labels = extract_labels(f)


train_images = train_images / 255.0
test_images = test_images / 255.0


#"""
print("\n\n\n############# USING CONVOLUTION BEFORE REGULARIZATIONS #############")

model = keras.Sequential([keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
			keras.layers.MaxPooling2D((2, 2)),
			keras.layers.Conv2D(64, (3, 3), activation='relu'),
			keras.layers.MaxPooling2D((2, 2)),
			keras.layers.Conv2D(64, (3, 3), activation='relu'),
			keras.layers.MaxPooling2D((2, 2)),
			keras.layers.Flatten(),
			keras.layers.Dense(500,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001), bias_regularizer=keras.regularizers.l2(0.001)),
			keras.layers.Dense(500,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001), bias_regularizer=keras.regularizers.l2(0.001)),
			keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='sgd',loss=tensorflow.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
train_images_new = train_images.reshape(-1,28, 28, 1)
test_images_new = test_images.reshape(-1,28, 28, 1)
l2_conv_history=model.fit(train_images_new, train_labels, epochs=250,batch_size=240,validation_split=0.13)
loss,accuracy = model.evaluate(test_images_new,test_labels, verbose=2)
print('\n\n\nTest accuracy with convolution before l2 regularization:',accuracy)
#"""

#"""
print("\n\n\n############# USING CONVOLUTION BEFORE DROPOUT #############")

model = keras.Sequential([keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
			keras.layers.MaxPooling2D((2, 2)),
			keras.layers.Dropout(0.1),
			keras.layers.Conv2D(64, (3, 3), activation='relu'),
			keras.layers.MaxPooling2D((2, 2)),
			keras.layers.Dropout(0.2),
			keras.layers.Conv2D(64, (3, 3), activation='relu'),
			keras.layers.MaxPooling2D((2, 2)),
			keras.layers.Dropout(0.2),
			keras.layers.Flatten(),
			keras.layers.Dense(500, activation='relu'),
			keras.layers.Dropout(0.5),
			keras.layers.Dense(500,activation='relu'),
			keras.layers.Dropout(0.5),
			keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='sgd',loss=tensorflow.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
train_images_new = train_images.reshape(-1,28, 28, 1)
test_images_new = test_images.reshape(-1,28, 28, 1)
dropout_conv_history=model.fit(train_images_new, train_labels, epochs=250,batch_size=240,validation_split=0.13)
loss,accuracy = model.evaluate(test_images_new,test_labels, verbose=2)
print('\n\n\nTest accuracy with convolution before dropout:',accuracy)
#"""


#"""
print("\n\n\n############# USING REGULARIZATIONS #############")

model = keras.Sequential([keras.layers.Flatten(input_shape=train_images[0].shape),
			keras.layers.Dense(500, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001), bias_regularizer=keras.regularizers.l2(0.001)),
			keras.layers.Dense(500,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001), bias_regularizer=keras.regularizers.l2(0.001)),
			keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='sgd',loss=tensorflow.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
l2_history=model.fit(train_images, train_labels, epochs=250,batch_size=240,validation_split=0.13)
loss,accuracy = model.evaluate(test_images,test_labels, verbose=2)
print('\n\n\nTest accuracy with l2 regularization:',accuracy)
#"""


#"""
print("\n\n\n############# USING DROPOUT #############")

model = keras.Sequential([keras.layers.Flatten(input_shape=train_images[0].shape),
			keras.layers.Dropout(0.2),
			keras.layers.Dense(500, activation='relu'),
			keras.layers.Dropout(0.5),
			keras.layers.Dense(500,activation='relu'),
			keras.layers.Dropout(0.5),
			keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='sgd',loss=tensorflow.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
dropout_history=model.fit(train_images, train_labels, epochs=250,batch_size=240,validation_split=0.13)
loss,accuracy = model.evaluate(test_images,test_labels, verbose=2)
print('\n\n\nTest accuracy with l2 regularization:',accuracy)
#"""


#"""
plt.figure(figsize=(50,20))
plt.subplot(2,4,1)
plt.plot(l2_history.history['acc'])
plt.plot(l2_history.history['val_acc'])
plt.title('accuracy vs epoch (with l2 regularization)',fontsize=8)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(2,4,2)
plt.plot(dropout_history.history['acc'])
plt.plot(dropout_history.history['val_acc'])
plt.title('accuracy vs epoch (with dropout)',fontsize=8)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(2,4,3)
plt.plot(l2_history.history['loss'])
plt.plot(l2_history.history['val_loss'])
plt.title('classification error vs epoch (with l2 regularization)',fontsize=8)
plt.ylabel('classification error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(2,4,4)
plt.plot(dropout_history.history['loss'])
plt.plot(dropout_history.history['val_loss'])
plt.title('classification error vs epoch (with dropout)',fontsize=8)
plt.ylabel('classification error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(2,4,5)
plt.plot(l2_conv_history.history['acc'])
plt.plot(l2_conv_history.history['val_acc'])
plt.title('accuracy vs epoch (with convolution before l2 regularization)',fontsize=6)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(2,4,6)
plt.plot(dropout_conv_history.history['acc'])
plt.plot(dropout_conv_history.history['val_acc'])
plt.title('accuracy vs epoch (with convolution before dropout)',fontsize=6)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(2,4,7)
plt.plot(l2_conv_history.history['loss'])
plt.plot(l2_conv_history.history['val_loss'])
plt.title('classification error vs epoch (with convolution before l2 regularization)',fontsize=6)
plt.ylabel('classification error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(2,4,8)
plt.plot(dropout_conv_history.history['loss'])
plt.plot(dropout_conv_history.history['val_loss'])
plt.title('classification error vs epoch (with convolution before dropout)',fontsize=6)
plt.ylabel('classification error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#plt.savefig("A04mpm514Q2.png")
plt.clf()
#"""
