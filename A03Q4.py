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

train_images = train_images.reshape((train_images.shape[0],IMG_SIZE*IMG_SIZE)).astype('float32')
test_images=test_images.reshape((test_images.shape[0],IMG_SIZE*IMG_SIZE)).astype('float32')
train_images = train_images / 255.0
test_images = test_images / 255.0
test_labels_new=convert(test_labels)
train_labels_new=convert(train_labels)


print("\n\n\n############# USING REGULARIZATIONS #############")
model = keras.Sequential([keras.layers.Dense(500, activation='relu',input_shape=train_images[0].shape,kernel_regularizer=keras.regularizers.l2(0.001), bias_regularizer=keras.regularizers.l2(0.001)),keras.layers.Dense(500,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001), bias_regularizer=keras.regularizers.l2(0.001)),keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='sgd',loss=tensorflow.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
l2_history=model.fit(train_images, train_labels, epochs=250,validation_split=0.13, batch_size=240)

loss,accuracy = model.evaluate(test_images,test_labels, verbose=2)
print('\n\n\nTest accuracy with l2 regularization:',accuracy)

print("\n\n\n############# USING DROPOUT #############")

model = keras.Sequential([keras.layers.Dropout(0.2,input_shape=train_images[0].shape),keras.layers.Dense(500, activation='relu'),keras.layers.Dropout(0.5),keras.layers.Dense(500,activation='relu'),keras.layers.Dropout(0.5),keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='sgd',loss=tensorflow.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
dropout_history=model.fit(train_images, train_labels, epochs=250,validation_split=0.13, batch_size=240)

loss,accuracy = model.evaluate(test_images,test_labels, verbose=2)
print('\n\n\nTest accuracy with dropout:',accuracy)

print("\n\n\n############# USING EARLY STOPPING #############")

model = keras.Sequential([keras.layers.Dense(500, activation='relu',input_shape=train_images[0].shape),keras.layers.Dense(500,activation='relu'),keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='sgd',loss=tensorflow.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
early_stopping=keras.callbacks.EarlyStopping(monitor="val_loss",mode="min",verbose=1)
es_history=model.fit(train_images, train_labels, epochs=250,validation_split=0.13, batch_size=240, callbacks=[early_stopping])

loss,accuracy = model.evaluate(test_images,test_labels, verbose=2)
print('\n\n\nTest accuracy with early stopping:',accuracy)

plt.figure(figsize=(25,15))
plt.subplot(3,3,1)
plt.plot(l2_history.history['acc'])
plt.plot(l2_history.history['val_acc'])
plt.title('accuracy vs epoch (with l2 regularization)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(3,3,2)
plt.plot(dropout_history.history['acc'])
plt.plot(dropout_history.history['val_acc'])
plt.title('accuracy vs epoch (with dropout)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(3,3,3)
plt.plot(es_history.history['acc'])
plt.plot(es_history.history['val_acc'])
plt.title('accuracy vs epoch (with early stopping)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(3,3,4)
plt.plot(l2_history.history['loss'])
plt.plot(l2_history.history['val_loss'])
plt.title('classification error vs epoch (with l2 regularization)')
plt.ylabel('classification error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(3,3,5)
plt.plot(dropout_history.history['loss'])
plt.plot(dropout_history.history['val_loss'])
plt.title('classification error vs epoch (with dropout)')
plt.ylabel('classification error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(3,3,6)
plt.plot(es_history.history['loss'])
plt.plot(es_history.history['val_loss'])
plt.title('classification error vs epoch (with early stopping)')
plt.ylabel('classification error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("A03mpm514Q4.png")
plt.clf()