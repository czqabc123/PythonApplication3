import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#x = np.linspace(-2, 6, 200)
x = np.arange(-100,200,1)
np.random.shuffle(x)
#y = 2 *x *x *x - 5 *x *x + 4 *x -5            2*x^3+5*x^2+4x-5
#y = - 5 *x *x + 4 *x -5                 aim
y = x**2+np.random.randn(300)*0.05
x_train,y_train = x[:200],y[:200]
#x_test ,y_test  = x[100:],y[100:]
plt.scatter(x,y)
plt.show()
#model =tf.keras.Sequential()
#model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=(1,),activation="elu"),
    tf.keras.layers.Dense(1)
])
model.summary()
model.compile(optimizer='adam',loss='mse')
model.fit(x,y,epochs=5000)
y_pred = model.predict(x_train)
plt.scatter(x_train, y_train)
plt.plot(x_train, y_pred,"r")
plt.show()
