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


#flattening the 28 x 28 images into a single 784 dimension vector
train_images = train_images.reshape((train_images.shape[0],IMG_SIZE*IMG_SIZE)).astype('float32')
test_images=test_images.reshape((test_images.shape[0],IMG_SIZE*IMG_SIZE)).astype('float32')
#normalizing the image intensity values
train_images = train_images / 255.0
test_images = test_images / 255.0
#converting the labels into one-hot encoded form
test_labels_new=convert(test_labels)
train_labels_new=convert(train_labels)

model = keras.Sequential([keras.layers.Dense(500, activation='relu',input_shape=train_images[0].shape),keras.layers.Dense(500,activation='relu'),keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='sgd',loss=tensorflow.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
history=model.fit(train_images, train_labels, epochs=250,validation_split=0.13, batch_size=240)

#plotting the graphs
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('accuracy vs epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss vs epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("A03mpm514Q3.png")
plt.clf()


#testing the model
loss,accuracy = model.evaluate(test_images,test_labels, verbose=2)
print('\nTest accuracy:',accuracy)
