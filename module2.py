import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 
import os
import sys
import time
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout,Flatten


(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
print('训练集共有{}个样本'.format(len(x_train)))
print('测试集共有{}个样本'.format(len(x_test)))
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)
model = tf.keras.Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
checkpointer =tf.keras.callbacks.ModelCheckpoint(filepath='mnist.model.best.hdf5',verbose =1,save_base_only=True)
hist = model.fit(x_train,y_train,batch_size =128,epochs =10,validation_split=0.2,callbacks=[checkpointer],verbose = 1,shuffle = True)
def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training','Validation'])

#    plt.figure()
#    plt.xlabel('Epochs')
#    plt.ylabel('Accuracy')
#    plt.plot(network_history.history['acc'])
#    plt.plot(network_history.history['val_acc'])
#    plt.legend(['Training','Validation'],loc ='lower right')
    plt.show()
plot_history(hist)