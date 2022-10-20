# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/18 13:53
@File: newton.py
@Desc: 
"""

def func(x):
    return x ** 3 - 2


def dfunc(x):
    return 3 * x ** 2


def Newton_3(c, t):
    while abs(func(t)) > 1e-6:
        t -= func(t) / dfunc(t)
        print(t)
    return t


print(Newton_3(2, 1))