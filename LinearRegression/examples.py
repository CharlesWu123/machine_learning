# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/27 13:53
@File: examples.py
@Desc: 
"""
import matplotlib.pyplot as plt
import numpy as np


# Plot 为绘图函数，同学们可以利用这个函数建立画布和基本的网格
def Plot():
    plt.figure()
    plt.title('Data')
    plt.xlabel('Diameter(Inches)')
    plt.ylabel('Price(Dollar)')
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt


plt = Plot()

# X 为披萨的直径列表，Y 为披萨的价格列表
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

# plt.plot(X, y, 'k.')
# plt.show()

from sklearn.linear_model import LinearRegression

# 一个线性回归模型对象model，此时model内的theta参数并没有值
model = LinearRegression()

# 数据预处理
X = np.array(X).reshape(-1,1)
y = np.array(y).reshape(-1,1)

# 利用model.fit（X, y），X为自变量，y为因变量
# 执行这一步之后，model中的 theta参数将变为基于输入的X, y在OLS模型下获得的训练值
model.fit(X, y)

# 对尺寸为12英寸的披萨价格进行预测
X_pre = [12]
X_pre = np.array(X_pre).reshape(-1,1)
print('匹萨价格预测值：$%.2f' % model.predict(X_pre)[0])