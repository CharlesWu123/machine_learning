# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/27 14:54
@File: gd.py
@Desc: 
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

buses = np.array(buses).reshape(-1, 1)
pgdp = np.array(pgdp).reshape(-1, 1)
buses_train = np.array(buses[:-3]).reshape(-1, 1)
pgdp_train = np.array(pgdp[:-3]).reshape(-1, 1)
buses_test = np.array(buses[-3:]).reshape(-1, 1)
pgdp_test = np.array(pgdp[-3:]).reshape(-1, 1)

# 对数据标准化
buses_max = np.max(buses)
buses_min = np.min(buses)
pgdp_max = np.max(pgdp)
pgdp_min = np.min(pgdp)

def data_norm(data, min, max):
    return (data - min) / (max - min)

def data_origin(data, min, max):
    return np.multiply((max-min), data) + min

# 训练集
buses_train = data_norm(buses_train, buses_min, buses_max)
pgdp_train = data_norm(pgdp_train, pgdp_min, pgdp_max)
# 测试集
buses_test = data_norm(buses_test, buses_min, buses_max)
pgdp_test = data_norm(pgdp_test, pgdp_min, pgdp_max)


def get_pgdp(theta, buses):
    # 给定 buses 来得到 pgdp
    pred = []
    for i in range(len(buses)):
        pred.append(theta[0] + theta[1] * buses[i])
    return pred

def loss(y, y_hat):
    # 计算真实值和预测值之间的距离
    loss = 0
    for i in range(len(y)):
        loss += (y[i] - y_hat[i]) ** 2
    return loss / (len(y) * 2)

def dloss(y, y_hat, x):
    dx = 0
    for i in range(len(y)):
        dx += (y[i] - y_hat[i]) * x[i]
    return dx

def init_theta():
    # 初始化 theta
    theta = []
    theta.append(0)
    theta.append(np.random.normal(0, 1))
    return theta

def gradientDescent(train_buses, train_pgdp, test_buses, test_pgdp, epoches, lr):
    theta = init_theta()
    print(theta)
    for i in range(epoches):
        train_pred = get_pgdp(theta, train_buses)
        train_loss = loss(train_pgdp, train_pred)
        theta[0] += lr * dloss(train_pgdp, train_pred, [1] * len(buses))
        theta[1] += lr * dloss(train_pgdp, train_pred, train_buses)
        test_pred = get_pgdp(theta, test_buses)
        test_loss = loss(test_pgdp, test_pred)
        if i % 1000 == 0:
            print("【{}】train loss: {:.6f}, test loss: {:.6f}, theta: [{}, {}]".format(
                i, float(train_loss), float(test_loss), theta[0], theta[1]))
    return theta

epoches = 10000
lr = 1e-3

theta = gradientDescent(buses_train, pgdp_train, buses_test, pgdp_test, epoches, lr)
print(theta)
# theta = [np.array([-0.05343621]), np.array([1.12797199])]
# 预测绘制
buses_pre_ori = [300000, 400000, 500000]
buses_pre_ori = np.array(buses_pre_ori).reshape(-1, 1)
buses_pre_norm = data_norm(buses_pre_ori, buses_min, buses_max)
pgdp_pre = get_pgdp(theta, buses_pre_norm)
pgdp_pre = data_origin(pgdp_pre, pgdp_min, pgdp_max)
# 绘制pgdp与buses之间的关系
plt = Plot()
plt.plot(buses, pgdp, 'k.')
plt.plot(buses_pre_ori, pgdp_pre, 'g-')
plt.show()