import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#x = np.linspace(-1,1,100)
x = np.arange(-100,200,1)
#np.random.shuffle(x)
#y = x**2+np.random.randn(100)*0.05
y = - 5 *x *x + 4 *x -5

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=(1,),activation="elu"),
    tf.keras.layers.Dense(1)
])


model.compile(
    optimizer="adam",
    loss="mse"
)


history = model.fit(x,y,epochs=5000)
y_predict = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_predict,'r')
plt.show()
