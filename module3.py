import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#拟合曲线

#x = np.linspace(-1,1,100)
x = np.arange(-30,60,0.1)
#y = - 5 *x *x *x + 4 *x*x -5 * x +8 - np.full(1000,1000000) * np.sin( 0.05*x )
y = x * x *0.8+ np.full(900,100) * np.sin(x)#0.8x^2+100*sinx
plt.scatter(x,y)
plt.plot(x,y,'r')
plt.show()
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50,input_shape=(1,),activation="elu"),#其中10是神经元数量，由于输入的是一维张量，故inshape是1
    tf.keras.layers.Dense(50,input_shape=(1,),activation="elu"),
    tf.keras.layers.Dense(50,input_shape=(1,),activation="elu"),
    tf.keras.layers.Dense(1)
])
model.compile(
    optimizer="adam",
    loss="mse"
)
history = model.fit(x,y,epochs=100000)#训练轮数8W
y_predict = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_predict,'r')
plt.show()
