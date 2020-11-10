import tensorflow as  tf
mnist = tf.keras.datasets.mnist
from tensorflow.keras.layers import Dense,Dropout,Flatten
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0,x_test/255.0
model = tf.keras.Sequential([Flatten(),
                             tf.keras.Dense(128,activation='relu'),
                             tf.keras.Dense(128,activation='softmax')
                             ])
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCate)