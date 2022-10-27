# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/27 14:50
@File: lr.py
@Desc: 
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
# 导入数据集
data = pd.read_csv('./data.csv')
print(data)

# buses: 城镇公交车运营数量
buses = data['Bus']
# pdgp: 人均国民生产总值
pgdp = data['PGDP']

#Plot 为绘图函数，同学们可以利用这个函数建立画布和基本的网格
def Plot():
    plt.figure()
    plt.title('Data')
    plt.xlabel('Buses')
    plt.ylabel('PGDP(Yuan)')
    plt.grid(True)
    return plt

# # 绘制pgdp与buses之间的关系
# plt = Plot()
# plt.plot(buses, pgdp, 'k.')
# plt.show()

# 1996-2004 训练
buses_train = np.array(buses[:-3]).reshape(-1, 1)
pgdp_train = np.array(pgdp[:-3]).reshape(-1, 1)
buses_test = np.array(buses[-3:]).reshape(-1, 1)
pgdp_test = np.array(pgdp[-3:]).reshape(-1, 1)
model = LinearRegression()
model.fit(buses_train, pgdp_train)

# 预测绘制
buses_pre = [300000, 400000, 500000]
buses_pre = np.array(buses_pre).reshape(-1, 1)
pgdp_pre = model.predict(buses_pre)
# 2005-2007预测
pgdp_test_pre = model.predict(buses_test)


# 绘制pgdp与buses之间的关系
plt = Plot()
plt.plot(buses, pgdp, 'k.')
plt.plot(buses_test, pgdp_test_pre, 'r.')
plt.plot(buses_pre, pgdp_pre, 'g-')
plt.show()
