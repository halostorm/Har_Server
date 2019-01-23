import matplotlib.pyplot as plt
import math
import numpy as np


def readAtt(FilePath):
    ekfX = []
    ekfY = []
    ekfZ = []

    gdfX = []
    gdfY = []
    gdfZ = []

    errorX = []
    errorY = []
    errorZ = []
    with open(FilePath, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if (len(line) < 15):
                continue
            ekfQ = []
            ekfQ.append(float(line[11]))
            ekfQ.append(float(line[12]))
            ekfQ.append(float(line[13]))
            ekfQ.append(float(line[14]))
            ekfEuler = Q2Euler(ekfQ)
            ekfX.append(ekfEuler[0])
            ekfY.append(ekfEuler[1])
            ekfZ.append(ekfEuler[2])

            gdfQ = []
            gdfQ.append(float(line[15]))
            gdfQ.append(float(line[16]))
            gdfQ.append(float(line[17]))
            gdfQ.append(float(line[18]))
            gdfEuler = Q2Euler(gdfQ)

            gdfX.append(gdfEuler[0])
            gdfY.append(gdfEuler[1])
            gdfZ.append(gdfEuler[2])

            errorX.append((gdfEuler[0] - ekfEuler[0]) / 5)
            errorY.append((gdfEuler[1] - ekfEuler[1]) / 5)
            erz = (gdfEuler[2] - ekfEuler[2]) / 5
            if np.abs(gdfEuler[2] - ekfEuler[2]) > 50:
                erz = 0
            errorZ.append(erz)

    return ekfX, ekfY, ekfZ, gdfX, gdfY, gdfZ, errorX, errorY, errorZ


def Q2Euler(q):
    eulerAngles = []
    eulerAngles.append(
        (float)(math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])) / np.pi * 180))
    eulerAngles.append((float)(
        math.asin(2.0 * (q[0] * q[2] - q[3] * q[1])) / np.pi * 180))
    eulerAngles.append((float)(
        math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])) / np.pi * 180))
    return eulerAngles


if __name__ == '__main__':
    filePath = r'/home/halo/attitude.txt'
    ekfX, ekfY, ekfZ, gdfX, gdfY, gdfZ, errorX, errorY, errorZ = readAtt(filePath)



    print((np.sqrt(np.mean(np.square(errorX)))))
    print((np.sqrt(np.mean(np.square(errorX)))))
    print((np.sqrt(np.mean(np.square(errorY)))))

    errorTrue = np.zeros(len(errorX))
    L = len(ekfX)
    x = np.linspace(0, 100, L)

    figEKF = plt.figure()
    plt.plot(x, errorX, lw=1, c='r' )
    plt.plot(x, errorY, lw=1, c='b' )
    plt.plot(x, errorZ, lw=1, c='g' )
    plt.plot(x, errorTrue, 'black', ls='-.')
    plt.legend()  # 展示图例
    plt.xlabel('Times/(seconds)')  # 给 x 轴添加标签
    plt.ylabel('error/(degrees)')  # 给 y 轴添加标签
    my_y_ticks = np.arange(-3, 5, 1)
    plt.yticks(my_y_ticks)

    # plt.plot(x, ekfZ, lw=1, c='r', label='Orientation Estimation from IMU',alpha = 0.9)
    # plt.plot(x, gdfZ, 'black', label='True', ls='-.',alpha = 0.9)
    #
    # plt.legend()  # 展示图例
    # plt.xlabel('Times/(seconds)')  # 给 x 轴添加标签
    # plt.ylabel('Yaw /(degrees)')  # 给 y 轴添加标签
    # plt.ylim(-180.00, 180.00)



    plt.show()
