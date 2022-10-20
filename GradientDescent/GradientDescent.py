# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/18 11:41
@File: GradientDescent.py
@Desc: 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy import genfromtxt
dataPath = r"./data1.csv"
dataSet = pd.read_csv(dataPath, header=None)
print(dataSet)
price = []
rooms = []
area = []
for data in range(0, len(dataSet)):
    area.append(dataSet[0][data])
    rooms.append(dataSet[1][data])
    price.append(dataSet[2][data])


def get_price(theta, area, rooms):
    # 给定 theta, area, rooms 来得到房价
    pred = []
    for i in range(len(area)):
        pred.append(theta[0] + theta[1] * area[i] + theta[2] * rooms[i])
    return pred


def loss_fun(y, y_hat):
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
    for i in range(3):
        if i == 0:
            theta.append(0)
        else:
            theta.append(np.random.normal(0, 1))
    return theta


def gradientDescent(rooms, price, area, epochs, lr):
    # 梯度下降求解
    theta = init_theta()
    print(theta)
    i = 0
    es_num = 0
    while True:
        pred = get_price(theta, area, rooms)
        loss = loss_fun(price, pred)
        theta0_dx = lr * dloss(price, pred, [1] * len(rooms))
        theta1_dx = lr * dloss(price, pred, area)
        theta2_dx = lr * dloss(price, pred, rooms)
        print(f'【{i}】{loss:.6f}, theta_dx: {theta0_dx}, {theta1_dx}, {theta2_dx}, '
              f'theta: {theta[0]:.6f}, {theta[1]:.6f}, {theta[2]:.6f}')
        # 当有连续五个epoch，theta的变化雄安与 1e-6时停止
        if (abs(theta0_dx) < 1e-6 and abs(theta1_dx) < 1e-6 and abs(theta2_dx) < 1e-6 and es_num>=5) or i > epochs:
            es_num += 1
            break
        elif not (abs(theta0_dx) < 1e-6 and abs(theta1_dx) < 1e-6 and abs(theta2_dx) < 1e-6):
            es_num = 0
        theta[0] += theta0_dx
        theta[1] += theta1_dx
        theta[2] += theta2_dx
        i += 1
    return theta


def draw(x, y, z, theta=[]):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c='r')
    if theta:
        area = np.linspace(1000, 3000, 1000)
        rooms = np.linspace(1, 5, 1000)
        price = get_price(theta, area, rooms)
        ax.plot(area, rooms, price)

    # 绘制图例
    ax.legend(loc='best')
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_xlabel('area', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('rooms', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('price', fontdict={'size': 15, 'color': 'red'})
    plt.show()


if __name__ == '__main__':
    epochs = 50000
    lr = 1e-8
    theta = gradientDescent(rooms, price, area, epochs, lr)
    print(theta)

    # theta = [-1.228134342799867, 0.18303219716102045, 0.755333000328144]
    # draw(area, rooms, price, theta)