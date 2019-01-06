# polynomial fitting
import numpy as np
import matplotlib.pyplot as plt
x_real = np.random.randint(100,size=1000)
x = np.array([0, 1, 2, 3, 4, 5, 6,8,9,10,11,12,13,14,15])
#x_real = np.array(
#    [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
y = np.array(2 * (x_real ** 4) + x_real ** 2 + 9 * x_real + 2)

# y = np.array([300,500,0,-10,0,20,200,300,1000,800,4000,5000,10000,9000,22000])

# coef 为系数，poly_fit 拟合函数
window_length = 15
exponent = 2
plt.axis([0, window_length, 0, 10000])
plt.ion()
for i in range(len(y)):
    plt.clf()
    window_x = x
    window_y = y[i:i + window_length]
    coef = np.polyfit(window_x, window_y, exponent)
    poly_fit = np.poly1d(coef)


    plt.plot(window_x, poly_fit(window_x), 'b')
    print(poly_fit)
    plt.pause(0.1)
