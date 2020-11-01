import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt	
#num_points=1000    
#vectors_set=[]
#def inputs():
#	for i in range(num_points):
#		x1=np.random.normal(0.0,0.55)   
#		y1=x1*0.1+0.3+np.random.normal(0.0,0.03)
#		vectors_set.append([x1,y1])
#	x_data=[v[0] for v in vectors_set]
#	y_data=[v[1] for v in vectors_set]
#	return x_data, y_data
#x_vals, y_vals = inputs()
x_ver= np.array([2,  3,  4,   5,6,7,8,9,10,11,12,13,14])
y_ver= np.array([4,5.5,7.9,10.2,12.3,14.1,16.3,18.5,19.2,22.3,24.3,26.5,28.1])
print(x_ver,y_ver)
#plt.plot(x_ver, y_ver, 'b.')
#plt.show()

model = keras.Sequential([layers.Dense(1, activation='sigmoid', input_shape=(1,))])

# 配置模型
model.compile(optimizer=keras.optimizers.SGD(0.1),#随机梯度下降法0.1  1：学习率，2：动量参数，3学习衰减率4是否支持支持Nesterov动量
             loss='mean_squared_error',  # keras.losses.mean_squared_error
             metrics=['mse'])#均
model.summary()#输出计算参数
model.fit(x_ver, y_ver,epochs=10)
model.predict(1)