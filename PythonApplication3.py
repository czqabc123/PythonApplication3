import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from matplotlib import pyplot as plt
print(tf.__version__)
# 导入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
print(x_train.shape, ' ', y_train.shape) 
print("XXXXXXXXXXXXXXXXX")
print(x_test.shape, ' ', y_test.shape)
print("YYYYYYYYYYYYYYYYYYY")
print(x_train,y_train)
plt.plot(x_train,y_train,color = "red")
plt.show()
# 构建模型

model = keras.Sequential([layers.Dense(32, activation='sigmoid', input_shape=(13,)),layers.Dense(32, activation='sigmoid'),layers.Dense(32, activation='sigmoid'),layers.Dense(1)])

# 配置模型
model.compile(optimizer=keras.optimizers.SGD(0.1),#随机梯度下降法0.1  1：学习率，2：动量参数，3学习衰减率4是否支持支持Nesterov动量
             loss='mean_squared_error',  # keras.losses.mean_squared_error
             metrics=['mse'])#均方根误差
model.summary()#输出计算参数
# 训练
model.fit(x_train, y_train, batch_size=50, epochs=50, validation_split=0.1, verbose=1)#verbose：输出日志的选项      validation_split用于在没有提供验证集的时候，按一定比例从训练集中取出一部分作为验证集
#epochs 轮数    batch_size 一次训练所选取的样本数。 
result = model.evaluate(x_test, y_test)#显示结果
print(model.metrics_names)
print(result)
print(model.predict(x_test))